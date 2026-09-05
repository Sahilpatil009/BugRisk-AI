import asyncio
import logging
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from .analyzer import analyze_repository
from .auth import decrypt_token
from .config import get_settings
from .database import SessionLocal
from .github_app import publish_pull_request_report
from .models import (
    Analysis,
    AnalysisStatus,
    FileResult,
    ModelExplanation,
    Prediction,
    PullRequest,
    Recommendation,
    utcnow,
)
from .recommendation_rewriter import rewrite_recommendations
from .risk import ModelService, file_priority, recommendations, risk_level

logger = logging.getLogger(__name__)


def process_analysis(analysis_id: str) -> None:
    settings = get_settings()
    db: Session = SessionLocal()
    try:
        claimed = cast(
            CursorResult[Any],
            db.execute(
                update(Analysis)
                .where(Analysis.id == analysis_id, Analysis.status == AnalysisStatus.QUEUED)
                .values(status=AnalysisStatus.ANALYZING)
            ),
        )
        if claimed.rowcount != 1:
            db.rollback()
            return
        db.commit()
        analysis = db.get(Analysis, analysis_id)
        if not analysis:
            return
        analysis.error_message = None
        db.commit()
        repository = analysis.repository
        pull_request = db.scalar(select(PullRequest).where(PullRequest.analysis_id == analysis.id))
        if pull_request:
            pull_request.status = AnalysisStatus.ANALYZING
            pull_request.updated_at = utcnow()
            db.commit()
        token = decrypt_token(repository_user_token(repository), settings)
        snapshot = analyze_repository(
            repository.url,
            repository.default_branch,
            token,
            analysis.commit_sha,
            changed_paths=pull_request.changed_files if pull_request else None,
            base_sha=pull_request.base_sha if pull_request else None,
            pull_request_number=pull_request.github_pr_number if pull_request else None,
        )
        analysis.status = AnalysisStatus.PREDICTING
        if pull_request:
            pull_request.status = AnalysisStatus.PREDICTING
            pull_request.updated_at = utcnow()
        analysis.commit_sha = snapshot.commit_sha
        db.commit()

        model = ModelService(settings.model_path)
        prediction = model.predict(snapshot.commit_features)
        priorities: list[float] = []
        for metrics in snapshot.files:
            score = file_priority(prediction.probability, metrics)  # type: ignore[arg-type]
            priorities.append(score)
            deterministic_actions = recommendations(
                prediction.explanations,
                metrics,  # type: ignore[arg-type]
            )
            actions = rewrite_recommendations(
                deterministic_actions, prediction.explanations, settings
            )
            recommendation_source = "llm" if actions != deterministic_actions else "deterministic"
            file_result = FileResult(
                analysis_id=analysis.id,
                file_path=str(metrics["file_path"]),
                file_priority_score=score,
                risk_level=risk_level(score),
                loc=int(metrics["loc"]),
                complexity=float(metrics["complexity"]),
                code_churn=int(metrics["code_churn"]),
                commit_count=int(metrics["commit_count"]),
                contributor_count=int(metrics["contributor_count"]),
                file_age_days=int(metrics["file_age_days"]),
                lines_added=int(metrics["lines_added"]),
                lines_deleted=int(metrics["lines_deleted"]),
                dependency_count=int(metrics["dependency_count"]),
                explanations=prediction.explanations,
                recommendations=actions,
            )
            db.add(file_result)
            db.flush()
            stored_prediction = Prediction(
                file_id=file_result.id,
                model_version=prediction.model_version,
                change_risk_probability=prediction.probability,
            )
            db.add(stored_prediction)
            db.flush()
            db.add_all(
                [
                    ModelExplanation(prediction_id=stored_prediction.id, **factor)
                    for factor in prediction.explanations
                ]
                + [
                    Recommendation(
                        prediction_id=stored_prediction.id,
                        text=action,
                        source=recommendation_source,
                    )
                    for action in actions
                ]
            )
        analysis.change_risk_probability = prediction.probability
        analysis.overall_priority_score = max(priorities, default=prediction.probability)
        analysis.risk_level = risk_level(prediction.probability)
        analysis.model_version = prediction.model_version
        analysis.status = AnalysisStatus.COMPLETED
        analysis.completed_at = utcnow()
        if pull_request:
            pull_request.status = AnalysisStatus.COMPLETED
            pull_request.risk_score = analysis.change_risk_probability
            pull_request.risk_level = analysis.risk_level
            pull_request.updated_at = utcnow()
        db.commit()
        if pull_request:
            try:
                comment_id = asyncio.run(
                    publish_pull_request_report(
                        settings, pull_request, analysis, list(analysis.files)
                    )
                )
                pull_request.github_comment_id = comment_id
                pull_request.updated_at = utcnow()
                db.commit()
            except Exception:
                logger.exception(
                    "pull_request_comment_failed",
                    extra={"analysis_id": analysis_id, "pull_request_id": pull_request.id},
                )
    except Exception as exc:
        logger.exception("analysis_failed", extra={"analysis_id": analysis_id})
        db.rollback()
        analysis = db.get(Analysis, analysis_id)
        if analysis:
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = str(exc)[:1000]
            analysis.completed_at = utcnow()
            pull_request = db.scalar(
                select(PullRequest).where(PullRequest.analysis_id == analysis.id)
            )
            if pull_request:
                pull_request.status = AnalysisStatus.FAILED
                pull_request.updated_at = utcnow()
            db.commit()
    finally:
        db.close()


def repository_user_token(repository) -> str | None:
    user = repository_user(repository)
    return user.encrypted_github_token if user else None


def repository_user(repository):
    from .models import User

    db = Session.object_session(repository)
    return db.get(User, repository.user_id) if db else None
