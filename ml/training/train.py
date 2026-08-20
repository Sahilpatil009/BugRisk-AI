import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.app.risk import CalibratedRiskModel
from ml.features.schema import FEATURE_NAMES


def temporal_project_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts, validation_parts, test_parts = [], [], []
    for _, project in frame.sort_values("event_time").groupby("project", sort=False):
        count = len(project)
        train_end = max(1, int(count * 0.6))
        validation_end = max(train_end + 1, int(count * 0.8))
        train_parts.append(project.iloc[:train_end])
        validation_parts.append(project.iloc[train_end:validation_end])
        test_parts.append(project.iloc[validation_end:])
    return tuple(
        pd.concat(parts, ignore_index=True)
        for parts in (train_parts, validation_parts, test_parts)
    )  # type: ignore[return-value]


def _pipeline(classifier, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [("numeric", Pipeline(steps), list(range(len(FEATURE_NAMES))))]
                ),
            ),
            ("classifier", classifier),
        ]
    )


def _metrics(
    labels: pd.Series, probability: np.ndarray
) -> dict[str, float | list[list[int]]]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
    }


def train(data_path: Path, output: Path) -> dict:
    frame = pd.read_csv(data_path, parse_dates=["event_time"])
    train_frame, validation_frame, test_frame = temporal_project_split(frame)
    if any(
        part.empty or part["buggy"].nunique() < 2
        for part in (train_frame, validation_frame, test_frame)
    ):
        raise ValueError("Each temporal split must contain buggy and clean samples")
    x_train, y_train = train_frame[FEATURE_NAMES].to_numpy(), train_frame["buggy"]
    x_validation, y_validation = (
        validation_frame[FEATURE_NAMES].to_numpy(),
        validation_frame["buggy"],
    )
    x_test, y_test = test_frame[FEATURE_NAMES].to_numpy(), test_frame["buggy"]
    candidates = {
        "logistic_regression": _pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True
        ),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
            )
        ),
        "xgboost": _pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        ),
    }
    validation_metrics = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        validation_metrics[name] = _metrics(
            y_validation, model.predict_proba(x_validation)[:, 1]
        )
    selected_name = max(
        candidates,
        key=lambda name: (
            validation_metrics[name]["pr_auc"],
            validation_metrics[name]["recall"],
        ),
    )
    selected = candidates[selected_name]
    validation_probability = np.clip(
        selected.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6
    )
    calibrator = LogisticRegression().fit(
        np.log(validation_probability / (1 - validation_probability)).reshape(-1, 1),
        y_validation,
    )
    calibrated = CalibratedRiskModel(selected, calibrator)
    test_metrics = _metrics(y_test, calibrated.predict_proba(x_test)[:, 1])
    model_version = f"apachejit-{selected_name}-v1"
    numeric_metrics = {
        key: value for key, value in test_metrics.items() if isinstance(value, float)
    }
    bundle = {
        "model": calibrated,
        "model_version": model_version,
        "model_name": selected_name,
        "feature_names": FEATURE_NAMES,
        "metrics": numeric_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "bugrisk_model.joblib")
    (output / "metrics.json").write_text(
        json.dumps(
            {key: value for key, value in bundle.items() if key != "model"}, indent=2
        ),
        encoding="utf-8",
    )
    try:
        import mlflow

        with mlflow.start_run(run_name=model_version):
            mlflow.log_params(
                {
                    "selected_model": selected_name,
                    "split": "60/20/20 temporal per project",
                }
            )
            mlflow.log_metrics(numeric_metrics)
            mlflow.log_artifact(str(output / "metrics.json"))
    except ImportError:
        pass
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
