import hashlib
import hmac
import json

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.database import SessionLocal
from app.github_app import render_pull_request_report, verify_webhook_signature
from app.main import app, settings
from app.models import Analysis, PullRequest, Repository


def _signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_webhook_signature_rejects_missing_and_tampered_payloads():
    body = b'{"action":"opened"}'
    assert verify_webhook_signature(body, _signature(body, "secret"), "secret")
    assert not verify_webhook_signature(body + b"x", _signature(body, "secret"), "secret")
    assert not verify_webhook_signature(body, None, "secret")


def test_pull_request_webhook_persists_changed_files_and_analysis(monkeypatch):
    with SessionLocal() as db:
        repository = db.scalar(select(Repository).limit(1))
        assert repository is not None
        repository_id = repository.id
        github_repo_id = repository.github_repo_id
        owner = repository.owner
        name = repository.name
        db.execute(
            delete(PullRequest).where(
                PullRequest.repository_id == repository_id,
                PullRequest.github_pr_number == 12,
            )
        )
        db.commit()

    async def token(*_args, **_kwargs):
        return "installation-token"

    async def files(*_args, **_kwargs):
        return ["src/risky.py", "README.md"]

    monkeypatch.setattr("app.main.installation_token", token)
    monkeypatch.setattr("app.main.pull_request_files", files)
    monkeypatch.setattr(settings, "github_webhook_secret", "webhook-secret")
    payload = {
        "action": "opened",
        "installation": {"id": 77},
        "number": 12,
        "repository": {"id": int(github_repo_id) if github_repo_id.isdigit() else github_repo_id},
        "pull_request": {
            "title": "Add risky change",
            "html_url": f"https://github.com/{owner}/{name}/pull/12",
            "state": "open",
            "user": {"login": "octocat"},
            "head": {"sha": "a" * 40},
            "base": {"sha": "b" * 40},
        },
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": "delivery-12",
        "X-Hub-Signature-256": _signature(body, "webhook-secret"),
        "Content-Type": "application/json",
    }
    with TestClient(app) as client:
        response = client.post("/webhooks/github", content=body, headers=headers)
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"
        duplicate = client.post("/webhooks/github", content=body, headers=headers)
        assert duplicate.json()["status"] == "duplicate"

    with SessionLocal() as db:
        item = db.scalar(
            select(PullRequest).where(
                PullRequest.repository_id == repository_id,
                PullRequest.github_pr_number == 12,
            )
        )
        assert item is not None
        assert item.changed_files == ["src/risky.py", "README.md"]
        assert item.analysis_id is not None
        assert db.get(Analysis, item.analysis_id) is not None


def test_invalid_webhook_signature_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "github_webhook_secret", "webhook-secret")
    with TestClient(app) as client:
        response = client.post(
            "/webhooks/github",
            content=b"{}",
            headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": "sha256=bad"},
        )
    assert response.status_code == 401


def test_pull_request_report_preserves_risk_semantics():
    class Level:
        value = "HIGH"

    class AnalysisStub:
        change_risk_probability = 0.71
        risk_level = Level()
        commit_sha = "a" * 40

    class FileStub:
        file_path = "src/risky.py"
        file_priority_score = 0.83
        risk_level = Level()
        explanations = [{"feature_name": "code_churn", "feature_value": 20, "shap_value": 0.2}]
        recommendations = ["Add regression coverage."]

    report = render_pull_request_report(object(), AnalysisStub(), [FileStub()])
    assert "Calibrated change risk" in report
    assert "review-priority rankings" in report
    assert "probabilities that a file contains a bug" in report
