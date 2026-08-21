from fastapi.testclient import TestClient

from app.main import app


def test_demo_story_exposes_repository_analysis_and_files():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        repositories = client.get("/repositories").json()
        assert repositories
        analyses = client.get("/analyses", params={"repository_id": repositories[0]["id"]}).json()
        completed = next(analysis for analysis in analyses if analysis["status"] == "COMPLETED")
        files = client.get(f"/analyses/{completed['id']}/files").json()
        assert files["total"] >= 5
        assert "file_priority_score" in files["items"][0]
        assert "change_risk_probability" not in files["items"][0]


def test_predict_validates_and_distinguishes_change_risk():
    payload = {
        "lines_added": 121,
        "lines_deleted": 47,
        "files_changed": 5,
        "code_churn": 168,
        "commit_entropy": 1.2,
        "developer_experience": 47,
        "file_age_days": 320,
        "commit_frequency": 14,
        "contributor_count": 4,
        "complexity": 21,
        "previous_defects": 4,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert 0 <= response.json()["change_risk_probability"] <= 1


def test_repository_connection_rejects_non_github_url():
    with TestClient(app) as client:
        response = client.post(
            "/repositories/connect",
            json={
                "github_repo_id": "x",
                "name": "x",
                "owner": "x",
                "url": "https://example.com/x.git",
            },
        )
        assert response.status_code == 422
