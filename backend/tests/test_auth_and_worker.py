from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import HTTPException
from sqlalchemy import select

from app.auth import begin_oauth, decrypt_token, encrypt_token
from app.config import Settings
from app.database import SessionLocal
from app.models import Analysis, AnalysisStatus, Repository
from app.recommendation_rewriter import rewrite_recommendations
from app.services import process_analysis


def test_token_encryption_round_trip():
    settings = Settings(session_secret="a-secure-test-secret-that-is-long-enough")
    encrypted = encrypt_token("github-token", settings)
    assert encrypted != "github-token"
    assert decrypt_token(encrypted, settings) == "github-token"


def test_oauth_refuses_missing_configuration():
    settings = Settings(
        session_secret="a-secure-test-secret-that-is-long-enough",
        github_client_id=None,
        github_client_secret=None,
    )
    try:
        begin_oauth(object(), settings)  # type: ignore[arg-type]
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("OAuth should require provider credentials")


def test_oauth_redirect_uses_callback_scope_and_session_state():
    class RequestStub:
        session: dict[str, str] = {}

    request = RequestStub()
    settings = Settings(
        session_secret="a-secure-test-secret-that-is-long-enough",
        github_client_id="configured-client",
        github_client_secret="configured-secret",
        backend_url="http://localhost:8000",
    )

    redirect = begin_oauth(request, settings)  # type: ignore[arg-type]
    query = parse_qs(urlparse(redirect).query)

    assert query["redirect_uri"] == ["http://localhost:8000/auth/github/callback"]
    assert query["scope"] == ["read:user user:email repo"]
    assert query["state"] == [request.session["oauth_state"]]


def test_llm_rewrite_has_deterministic_fallback(monkeypatch):
    def fail(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", fail)
    settings = Settings(
        session_secret="a-secure-test-secret-that-is-long-enough",
        llm_rewrite_url="https://provider.invalid/rewrite",
    )
    actions = ["Add regression coverage."]
    assert rewrite_recommendations(actions, [], settings) == actions


def test_failed_worker_transition_is_persisted(monkeypatch):
    with SessionLocal() as db:
        repository = db.scalar(select(Repository).limit(1))
        assert repository is not None
        analysis = Analysis(repository_id=repository.id)
        db.add(analysis)
        db.commit()
        analysis_id = analysis.id

    def fail_analysis(*_args, **_kwargs):
        raise RuntimeError("fixture clone failed")

    monkeypatch.setattr("app.services.analyze_repository", fail_analysis)
    process_analysis(analysis_id)

    with SessionLocal() as db:
        failed = db.get(Analysis, analysis_id)
        assert failed is not None
        assert failed.status == AnalysisStatus.FAILED
        assert failed.error_message == "fixture clone failed"
