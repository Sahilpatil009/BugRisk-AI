"""Phase 6 experiment: CodeBERT + metrics MLP vs. metrics-only XGBoost.

Both models see exactly the same rows (commits whose source code resolved in
the GitHub mirrors) and follow the same evaluation protocol as the deployed
baseline: per-project temporal 60/20/20 split, Platt calibration on
validation probabilities, and a recall-aware F2 threshold chosen on
validation.  The difference is the input: the baseline uses the 11 git/code
metrics while the CodeBERT model uses the 11 metrics plus a 768-d
mean-pooled CodeBERT vector of the commit's changed source files.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.features.schema import FEATURE_NAMES
from ml.training.train import (
    _metrics,
    _pipeline,
    select_operating_threshold,
    temporal_project_split,
)

CODEBERT_MODEL_VERSION = "apachejit-codebert-mlp-v1"


def load_experiment_frame(
    features_csv: Path, embeddings_npz: Path
) -> tuple[pd.DataFrame, dict]:
    features = pd.read_csv(features_csv, parse_dates=["event_time"])
    with np.load(embeddings_npz, allow_pickle=False) as bundle:
        commit_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in bundle["commit_id"]]
        embeddings = np.asarray(bundle["embeddings"], dtype=np.float32)
    embedding_frame = pd.DataFrame(
        {
            "commit_id": commit_ids,
            **{f"emb_{index}": embeddings[:, index] for index in range(embeddings.shape[1])},
        }
    )
    merged = features.merge(embedding_frame, on="commit_id", how="inner", validate="many_to_one")
    merged = merged.drop_duplicates(subset="commit_id").reset_index(drop=True)
    if merged["buggy"].nunique() < 2:
        raise ValueError("CodeBERT experiment needs both buggy and clean commits")
    return merged, {"embedding_dim": int(embeddings.shape[1])}


def _platt_calibrator(labels: pd.Series, probability: np.ndarray) -> LogisticRegression:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression().fit(logits, labels)


def _calibrated_probability(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _train_codebert_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    hidden_dim: int,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[dict, float]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    pos_weight = torch.tensor(float(len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1))
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_tensor = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    label_tensor = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    val_tensor = torch.from_numpy(np.ascontiguousarray(x_validation, dtype=np.float32))
    batch_size = 256
    best_state: dict | None = None
    best_score = -1.0
    stall = 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_tensor))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(train_tensor[batch]).squeeze(1)
            loss = loss_fn(logits, label_tensor[batch].squeeze(1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor).squeeze(1).numpy()
        val_probability = 1.0 / (1.0 + np.exp(-val_logits))
        score = float(average_precision_score(y_validation, val_probability))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_state is None:
        raise ValueError("CodeBERT MLP produced no improving epoch")
    return best_state, best_score


def train_codebert_experiment(
    data_path: Path,
    embeddings_path: Path,
    output: Path,
    hidden_dim: int = 256,
    epochs: int = 40,
    patience: int = 6,
    seed: int = 42,
) -> dict:
    frame, embedding_meta = load_experiment_frame(data_path, embeddings_path)
    embedding_columns = [column for column in frame.columns if column.startswith("emb_")]
    x_all = np.hstack(
        [frame[FEATURE_NAMES].to_numpy(dtype=np.float32), frame[embedding_columns].to_numpy(dtype=np.float32)]
    )
    train_frame, validation_frame, test_frame = temporal_project_split(frame)
    if any(part["buggy"].nunique() < 2 for part in (train_frame, validation_frame, test_frame)):
        raise ValueError("Each temporal split must contain buggy and clean samples")
    position = {commit: index for index, commit in enumerate(frame["commit_id"])}
    x_train = x_all[[position[commit] for commit in train_frame["commit_id"]]]
    x_validation = x_all[[position[commit] for commit in validation_frame["commit_id"]]]
    x_test = x_all[[position[commit] for commit in test_frame["commit_id"]]]
    y_train = train_frame["buggy"].to_numpy()
    y_validation = validation_frame["buggy"].to_numpy()
    y_test = test_frame["buggy"].to_numpy()

    metric_columns = [column for column in frame.columns if column in FEATURE_NAMES]
    scaler = StandardScaler().fit(frame[metric_columns].to_numpy(dtype=np.float32))
    x_train = np.hstack([scaler.transform(train_frame[metric_columns]), x_train[:, len(FEATURE_NAMES):]]).astype(np.float32)
    x_validation = np.hstack(
        [scaler.transform(validation_frame[metric_columns]), x_validation[:, len(FEATURE_NAMES):]]
    ).astype(np.float32)
    x_test = np.hstack([scaler.transform(test_frame[metric_columns]), x_test[:, len(FEATURE_NAMES):]]).astype(np.float32)

    baseline = _pipeline(
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
    )
    baseline.fit(x_train[:, : len(FEATURE_NAMES)], y_train)
    baseline_validation = baseline.predict_proba(x_validation[:, : len(FEATURE_NAMES)])[:, 1]
    baseline_calibrator = _platt_calibrator(y_validation, baseline_validation)
    baseline_calibrated_validation = _calibrated_probability(baseline_calibrator, baseline_validation)
    baseline_threshold = select_operating_threshold(y_validation, baseline_calibrated_validation)
    baseline_test = _metrics(
        y_test, _calibrated_probability(baseline_calibrator, baseline.predict_proba(x_test[:, : len(FEATURE_NAMES)])[:, 1]), baseline_threshold
    )

    state, validation_pr_auc = _train_codebert_mlp(
        x_train, y_train, x_validation, y_validation, hidden_dim=hidden_dim, epochs=epochs, patience=patience, seed=seed
    )
    import torch
    from torch import nn

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 1),
    )
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_validation)).squeeze(1).numpy()
        test_logits = model(torch.from_numpy(x_test)).squeeze(1).numpy()
    val_probability = 1.0 / (1.0 + np.exp(-val_logits))
    test_probability = 1.0 / (1.0 + np.exp(-test_logits))
    calibrator = _platt_calibrator(y_validation, val_probability)
    calibrated_val = _calibrated_probability(calibrator, val_probability)
    threshold = select_operating_threshold(y_validation, calibrated_val)
    codebert_test = _metrics(y_test, _calibrated_probability(calibrator, test_probability), threshold)

    delta: dict[str, float] = {}
    for name in ("precision", "recall", "f1", "f2", "roc_auc", "pr_auc"):
        codebert_value = codebert_test.get(name)
        baseline_value = baseline_test.get(name)
        if isinstance(codebert_value, float) and isinstance(baseline_value, float):
            delta[name] = round(codebert_value - baseline_value, 6)

    comparison = {
        "model_version": CODEBERT_MODEL_VERSION,
        "embedding_model": "codebert",
        "embedding_dim": embedding_meta["embedding_dim"],
        "config": {"hidden_dim": hidden_dim, "epochs": epochs, "patience": patience, "seed": seed},
        "split": "60/20/20 temporal per project (same protocol as the metrics baseline)",
        "subset": {
            "projects": sorted(frame["project"].unique().tolist()),
            "commits": len(frame),
            "commits_per_project": {
                str(project): int(count) for project, count in frame["project"].value_counts().items()
            },
            "buggy_rate": float(frame["buggy"].mean()),
            "validation_pr_auc_codebert": validation_pr_auc,
        },
        "baseline": {"model": "xgboost-metrics-only", "metrics": baseline_test},
        "codebert": {"model": "codebert-mlp", "metrics": codebert_test},
        "delta": delta,
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_version": CODEBERT_MODEL_VERSION,
            "torch_state": state,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_meta["embedding_dim"],
            "feature_names": FEATURE_NAMES,
            "scaler_mean": scaler.mean_.astype(np.float64),
            "scaler_scale": scaler.scale_.astype(np.float64),
            "calibrator": calibrator,
            "decision_threshold": threshold,
            "metrics": {key: value for key, value in codebert_test.items() if isinstance(value, float)},
        },
        output / "codebert_model.joblib",
    )
    (output / "codebert_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Processed ApacheJIT CSV with commit_id")
    parser.add_argument("--embeddings", type=Path, required=True, help="embeddings.npz from the embed stage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    comparison = train_codebert_experiment(
        args.data, args.embeddings, args.output, hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience
    )
    print(json.dumps({key: comparison[key] for key in ("baseline", "codebert", "delta")}, indent=2))


if __name__ == "__main__":
    main()
