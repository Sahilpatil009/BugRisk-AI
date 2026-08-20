from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any, TypedDict

import joblib  # type: ignore[import-untyped]
import numpy as np

from .models import RiskLevel

FEATURE_NAMES = [
    "lines_added",
    "lines_deleted",
    "files_changed",
    "code_churn",
    "commit_entropy",
    "developer_experience",
    "file_age_days",
    "commit_frequency",
    "contributor_count",
    "complexity",
    "previous_defects",
]


class ExplanationData(TypedDict):
    feature_name: str
    feature_value: float
    shap_value: float


def risk_level(score: float) -> RiskLevel:
    percent = score * 100
    if percent <= 30:
        return RiskLevel.LOW
    if percent <= 60:
        return RiskLevel.MEDIUM
    if percent <= 80:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def _clip_ratio(value: float, high: float) -> float:
    return min(max(value / high, 0.0), 1.0)


def file_priority(change_risk: float, metrics: dict[str, float]) -> float:
    exposure = (
        0.28 * _clip_ratio(metrics.get("code_churn", 0), 400)
        + 0.24 * _clip_ratio(metrics.get("complexity", 0), 30)
        + 0.18 * _clip_ratio(metrics.get("commit_count", 0), 25)
        + 0.12 * _clip_ratio(metrics.get("contributor_count", 0), 10)
        + 0.10 * _clip_ratio(metrics.get("dependency_count", 0), 15)
        + 0.08 * _clip_ratio(metrics.get("previous_defects", 0), 5)
    )
    return round(min(max(0.55 * change_risk + 0.45 * exposure, 0), 1), 4)


def recommendations(explanations: list[ExplanationData], metrics: dict[str, float]) -> list[str]:
    positive = {item["feature_name"] for item in explanations if item["shap_value"] > 0}
    actions: list[str] = []
    if "complexity" in positive or metrics.get("complexity", 0) >= 15:
        actions.append("Review conditional branches and add tests for complex paths.")
    if {"code_churn", "lines_added", "lines_deleted"} & positive or metrics.get(
        "code_churn", 0
    ) >= 150:
        actions.append("Review the highest-churn changes and add regression coverage.")
    if "previous_defects" in positive:
        actions.append("Retest scenarios associated with earlier defects in this area.")
    if "commit_frequency" in positive or metrics.get("commit_count", 0) >= 10:
        actions.append("Inspect recent commits for interacting or incomplete changes.")
    if metrics.get("dependency_count", 0) >= 8:
        actions.append("Run integration tests across the affected dependencies.")
    if not actions:
        actions.append("Run focused unit tests and complete a standard peer review.")
    return actions[:3]


@dataclass
class Prediction:
    probability: float
    explanations: list[ExplanationData]
    model_version: str


class CalibratedRiskModel:
    """Serializable sigmoid calibration wrapper around a fitted sklearn pipeline."""

    def __init__(self, base_model: Any, calibrator: Any):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        raw = np.clip(self.base_model.predict_proba(rows)[:, 1], 1e-6, 1 - 1e-6)
        logits = np.log(raw / (1 - raw)).reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(logits)[:, 1]
        return np.column_stack([1 - calibrated, calibrated])


class ModelService:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.bundle: dict[str, Any] | None = None
        if model_path.exists():
            loaded = joblib.load(model_path)
            if loaded.get("feature_names") != FEATURE_NAMES:
                raise ValueError("Model feature order does not match the inference schema")
            self.bundle = loaded

    def predict(self, features: dict[str, float]) -> Prediction:
        row = np.array([[float(features[name]) for name in FEATURE_NAMES]])
        if self.bundle:
            model = self.bundle["model"]
            probability = float(model.predict_proba(row)[0, 1])
            contributions = self._shap_values(model, row)
            version = str(self.bundle["model_version"])
        else:
            probability, contributions = self._demo_prediction(features)
            version = "demo-heuristic-v1"
        explanations: list[ExplanationData] = [
            {
                "feature_name": name,
                "feature_value": float(features[name]),
                "shap_value": round(float(value), 5),
            }
            for name, value in sorted(
                zip(FEATURE_NAMES, contributions, strict=True),
                key=lambda pair: abs(pair[1]),
                reverse=True,
            )[:5]
        ]
        return Prediction(round(probability, 4), explanations, version)

    @staticmethod
    def _shap_values(model: Any, row: np.ndarray) -> np.ndarray:
        try:
            import shap  # type: ignore[import-untyped]

            base_model = getattr(model, "base_model", model)
            if hasattr(base_model, "named_steps"):
                transformed = base_model.named_steps["preprocessor"].transform(row)
                estimator = base_model.named_steps["classifier"]
            else:
                transformed = row
                estimator = base_model
            values = shap.Explainer(estimator)(transformed).values
            if values.ndim == 3:
                values = values[:, :, 1]
            return np.asarray(values[0], dtype=float)
        except Exception:
            return np.zeros(len(FEATURE_NAMES), dtype=float)

    @staticmethod
    def _demo_prediction(features: dict[str, float]) -> tuple[float, np.ndarray]:
        scales = np.array([250, 150, 12, 400, 2.5, 100, 1000, 25, 10, 30, 5], dtype=float)
        weights = np.array([0.7, 0.5, 0.5, 1.0, 0.4, -0.35, -0.15, 0.8, 0.35, 1.0, 1.1])
        normalized = np.minimum(np.array([features[name] for name in FEATURE_NAMES]) / scales, 2)
        contributions = normalized * weights
        logit = -2.2 + float(contributions.sum())
        return 1 / (1 + exp(-logit)), contributions

    def metrics(self) -> dict[str, Any]:
        if self.bundle:
            return {
                "model_version": self.bundle["model_version"],
                "model_name": self.bundle["model_name"],
                "trained": True,
                "metrics": self.bundle.get("metrics", {}),
                "feature_names": FEATURE_NAMES,
            }
        return {
            "model_version": "demo-heuristic-v1",
            "model_name": "Transparent demo heuristic",
            "trained": False,
            "metrics": {},
            "feature_names": FEATURE_NAMES,
            "note": (
                "Train the ApacheJIT pipeline to replace the demo heuristic "
                "with a calibrated model."
            ),
        }
