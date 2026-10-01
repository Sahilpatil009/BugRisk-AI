import pytest

from app.models import RiskLevel
from app.risk import FEATURE_NAMES, ModelService, file_priority, recommendations, risk_level


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, RiskLevel.LOW),
        (0.30, RiskLevel.LOW),
        (0.3001, RiskLevel.MEDIUM),
        (0.60, RiskLevel.MEDIUM),
        (0.6001, RiskLevel.HIGH),
        (0.80, RiskLevel.HIGH),
        (0.8001, RiskLevel.CRITICAL),
        (1, RiskLevel.CRITICAL),
    ],
)
def test_risk_boundaries(score, expected):
    assert risk_level(score) == expected


def test_file_priority_is_bounded_and_not_named_probability():
    score = file_priority(0.8, {"code_churn": 10000, "complexity": 1000, "commit_count": 500})
    assert 0 <= score <= 1


def test_recommendations_follow_positive_evidence():
    items = recommendations(
        [{"feature_name": "complexity", "feature_value": 20, "shap_value": 0.5}],
        {"complexity": 20},
    )
    assert any("conditional branches" in item for item in items)


def test_demo_prediction_has_stable_contract(tmp_path):
    features = {name: 1.0 for name in FEATURE_NAMES}
    prediction = ModelService(tmp_path / "missing.joblib").predict(features)
    assert 0 <= prediction.probability <= 1
    assert prediction.model_version == "demo-heuristic-v1"
    assert len(prediction.explanations) == 5
