from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.risk import ModelService
from ml.features.schema import FEATURE_NAMES
from ml.training.train import train


def test_training_exports_loadable_calibrated_artifact(tmp_path: Path):
    rows = 120
    frame = pd.DataFrame(
        {
            feature: np.linspace(1, 50, rows) + (index % 3)
            for index, feature in enumerate(FEATURE_NAMES)
        }
    )
    frame["buggy"] = np.arange(rows) % 2
    frame["event_time"] = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    frame["project"] = [
        "alpha" if index < rows / 2 else "beta" for index in range(rows)
    ]
    data = tmp_path / "prepared.csv"
    output = tmp_path / "artifacts"
    frame.to_csv(data, index=False)

    bundle = train(data, output)
    service = ModelService(output / "bugrisk_model.joblib")
    prediction = service.predict({feature: 5.0 for feature in FEATURE_NAMES})

    assert bundle["model_name"] in {"logistic_regression", "random_forest", "xgboost"}
    assert 0.05 <= bundle["decision_threshold"] <= 0.5
    assert bundle["test_metrics"]["decision_threshold"] == bundle["decision_threshold"]
    assert 0 <= prediction.probability <= 1
    assert (output / "metrics.json").exists()
