from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import ensure_demo_user
from .models import Analysis, AnalysisStatus, FileResult, Repository, RiskLevel, utcnow

DEMO_FILES = [
    ("payments/service.py", 0.91, 532, 24, 344, 17, 6, 730, 241, 103, 9),
    ("checkout/orchestrator.py", 0.82, 411, 19, 289, 14, 5, 512, 190, 99, 7),
    ("database/session.py", 0.71, 298, 17, 201, 11, 4, 920, 121, 80, 6),
    ("notifications/email.py", 0.47, 220, 10, 108, 8, 3, 388, 70, 38, 5),
    ("auth/tokens.py", 0.23, 174, 5, 42, 3, 2, 1200, 28, 14, 3),
]


def seed_demo(db: Session) -> None:
    user = ensure_demo_user(db)
    repository = db.scalar(
        select(Repository).where(
            Repository.user_id == user.id, Repository.github_repo_id == "demo-101"
        )
    )
    if repository:
        return
    repository = Repository(
        user_id=user.id,
        github_repo_id="demo-101",
        name="ecommerce-backend",
        owner="bugrisk-demo",
        url="https://github.com/pallets/flask",
        default_branch="main",
    )
    db.add(repository)
    db.flush()
    analysis = Analysis(
        repository_id=repository.id,
        commit_sha="8f31d7e-demo",
        status=AnalysisStatus.COMPLETED,
        change_risk_probability=0.78,
        overall_priority_score=0.91,
        risk_level=RiskLevel.CRITICAL,
        model_version="demo-heuristic-v1",
        created_at=utcnow() - timedelta(hours=3),
        completed_at=utcnow() - timedelta(hours=3) + timedelta(seconds=18),
    )
    db.add(analysis)
    db.flush()
    for index, values in enumerate(DEMO_FILES):
        (
            path,
            score,
            loc,
            complexity,
            churn,
            commits,
            contributors,
            age,
            added,
            deleted,
            dependencies,
        ) = values
        explanations = [
            {
                "feature_name": "code_churn",
                "feature_value": churn,
                "shap_value": round(0.28 - index * 0.03, 3),
            },
            {
                "feature_name": "complexity",
                "feature_value": complexity,
                "shap_value": round(0.21 - index * 0.025, 3),
            },
            {
                "feature_name": "commit_frequency",
                "feature_value": commits,
                "shap_value": round(0.15 - index * 0.02, 3),
            },
        ]
        db.add(
            FileResult(
                analysis_id=analysis.id,
                file_path=path,
                file_priority_score=score,
                risk_level=RiskLevel.CRITICAL
                if score > 0.8
                else RiskLevel.HIGH
                if score > 0.6
                else RiskLevel.MEDIUM
                if score > 0.3
                else RiskLevel.LOW,
                loc=loc,
                complexity=complexity,
                code_churn=churn,
                commit_count=commits,
                contributor_count=contributors,
                file_age_days=age,
                lines_added=added,
                lines_deleted=deleted,
                dependency_count=dependencies,
                explanations=explanations,
                recommendations=[
                    "Review the highest-churn changes and add regression coverage.",
                    "Review conditional branches and add tests for complex paths.",
                    "Inspect recent commits for interacting or incomplete changes.",
                ],
            )
        )
    db.commit()
