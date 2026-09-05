import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from git import Actor, Repo

from ml.embeddings.embed import encode_sources
from ml.embeddings.extract import extract_from_repo


class _StubEncoder:
    """Deterministic fake CodeBERT: a fixed random vector per content hash."""

    def __init__(self, dim: int = 8):
        self.name = "stub-codebert"
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        vectors = np.vstack(
            [
                np.abs(
                    np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")).standard_normal(self.dim)
                )
                for text in texts
            ]
        ).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
        return vectors / norms


def _local_repo_with_commits(tmp_path: Path) -> tuple[Repo, list[str]]:
    repository = Repo.init(tmp_path)
    author = Actor("Researcher", "researcher@example.com")
    (tmp_path / "Payment.java").write_text("class Payment {\n}\n", encoding="utf-8")
    (tmp_path / "Notes.txt").write_text("not source\n", encoding="utf-8")
    repository.index.add(["Payment.java", "Notes.txt"])
    first = repository.index.commit("add payment", author=author, committer=author)
    (tmp_path / "Payment.java").write_text("class Payment {\n  void pay() {}\n}\n", encoding="utf-8")
    (tmp_path / "Auth.py").write_text("def check():\n    return True\n", encoding="utf-8")
    repository.index.add(["Payment.java", "Auth.py"])
    second = repository.index.commit("extend payment", author=author, committer=author)
    return repository, [first.hexsha, second.hexsha]


def test_extract_from_local_repo_captures_post_change_source(tmp_path: Path):
    _repository, (first_sha, second_sha) = _local_repo_with_commits(tmp_path)
    frame = pd.DataFrame(
        {
            "project": ["demo", "demo"],
            "commit_id": [first_sha, second_sha],
            "buggy": [0, 1],
            "event_time": pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC"),
        }
    )
    sources, result = extract_from_repo(frame, "demo", tmp_path, max_files=8, max_chars=1000)
    assert result.resolved_commits == 2
    assert result.commits_with_code == 2
    second = sources[sources["commit_id"] == second_sha]
    paths = set(second["file_path"])
    assert paths == {"Payment.java", "Auth.py"}
    payment = second[second["file_path"] == "Payment.java"].iloc[0]
    assert "void pay()" in payment["content"]
    assert payment["language"] == "java"
    # The non-source Notes.txt must never appear.
    assert not sources["file_path"].str.endswith(".txt").any()


def test_extract_respects_file_cap_and_missing_commits(tmp_path: Path):
    _repository, (first_sha, second_sha) = _local_repo_with_commits(tmp_path)
    frame = pd.DataFrame(
        {
            "project": ["demo", "demo", "demo"],
            "commit_id": [first_sha, second_sha, "f" * 40],
            "buggy": [0, 1, 0],
            "event_time": pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"),
        }
    )
    sources, result = extract_from_repo(frame, "demo", tmp_path, max_files=1, max_chars=1000)
    assert result.resolved_commits == 2
    assert result.total_commits == 3
    assert len(sources[sources["commit_id"] == second_sha]) == 1


def test_encode_sources_pools_files_and_caches(tmp_path: Path):
    encoder = _StubEncoder(dim=8)
    sources = pd.DataFrame(
        {
            "project": ["demo", "demo", "demo"],
            "commit_id": ["a1", "a1", "b2"],
            "file_path": ["A.java", "B.java", "C.py"],
            "language": ["java", "java", "python"],
            "content": ["alpha\n", "beta\n", "alpha\n"],  # C.py repeats A.java's content
        }
    )
    result = encode_sources(sources, tmp_path, encoder, tmp_path / "cache.sqlite3")
    assert result.commits_embedded == 2
    assert result.files_used == 3
    # C.py reuses A.java's cached encoding: only two unique contents were encoded.
    assert sum(len(batch) for batch in encoder.calls) == 2
    with np.load(tmp_path / "embeddings.npz", allow_pickle=False) as bundle:
        assert bundle["commit_id"].astype("U").tolist() == ["a1", "b2"]
        assert bundle["embeddings"].shape == (2, 8)
        for row in bundle["embeddings"]:
            assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-5)

    replay_encoder = _StubEncoder(dim=8)
    encode_sources(sources, tmp_path, replay_encoder, tmp_path / "cache.sqlite3")
    assert replay_encoder.calls == []  # fully served from cache


def test_train_codebert_experiment_writes_comparison(tmp_path: Path):
    pytest.importorskip("torch")
    from ml.training.train_codebert import train_codebert_experiment

    rows = 400
    rng = np.random.default_rng(7)
    metrics = rng.gamma(2.0, 10.0, size=(rows, 11)).astype(float)
    buggy = (rng.random(rows) < 0.3).astype(int)
    # Embeddings carry a signal correlated with the label so the MLP can learn.
    embedding = rng.normal(0, 1, size=(rows, 8)).astype(np.float32)
    embedding[buggy == 1, :2] += 3.0
    frame = pd.DataFrame(metrics, columns=[f"m{index}" for index in range(11)])
    frame = frame.rename(
        columns={
            "m0": "lines_added",
            "m1": "lines_deleted",
            "m2": "files_changed",
            "m3": "code_churn",
            "m4": "commit_entropy",
            "m5": "developer_experience",
            "m6": "file_age_days",
            "m7": "commit_frequency",
            "m8": "contributor_count",
            "m9": "complexity",
            "m10": "previous_defects",
        }
    )
    frame["buggy"] = buggy
    frame["event_time"] = pd.date_range("2015-01-01", periods=rows, freq="D", tz="UTC")
    frame["project"] = "demo"
    frame["commit_id"] = [f"{index:040x}" for index in range(rows)]
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    embeddings_path = tmp_path / "embeddings.npz"
    np.savez(
        embeddings_path,
        commit_id=frame["commit_id"].to_numpy(dtype="S40"),
        embeddings=embedding,
    )
    comparison = train_codebert_experiment(
        data_path, embeddings_path, tmp_path / "artifacts", hidden_dim=32, epochs=8, patience=3
    )
    assert set(comparison) >= {"baseline", "codebert", "delta", "subset"}
    assert comparison["subset"]["commits"] == rows
    for model in ("baseline", "codebert"):
        metrics_ = comparison[model]["metrics"]
        assert 0.0 <= metrics_["roc_auc"] <= 1.0
        assert "confusion_matrix" in metrics_
    artifact = tmp_path / "artifacts" / "codebert_comparison.json"
    assert json.loads(artifact.read_text(encoding="utf-8"))["model_version"].startswith("apachejit-codebert-mlp")
    assert (tmp_path / "artifacts" / "codebert_model.joblib").exists()
