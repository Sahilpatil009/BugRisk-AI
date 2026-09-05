from pathlib import Path

from git import Actor, Repo

from app.analyzer import _analyze_clone


def test_pull_request_analysis_only_returns_changed_python_paths(tmp_path: Path):
    repository = Repo.init(tmp_path)
    author = Actor("Tester", "tester@example.com")
    (tmp_path / "included.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "excluded.py").write_text("value = 1\n", encoding="utf-8")
    repository.index.add(["included.py", "excluded.py"])
    base = repository.index.commit("initial", author=author, committer=author)

    (tmp_path / "included.py").write_text("value = 1\nvalue = 2\n", encoding="utf-8")
    (tmp_path / "excluded.py").write_text("value = 1\nvalue = 3\n", encoding="utf-8")
    repository.index.add(["included.py", "excluded.py"])
    head = repository.index.commit("change", author=author, committer=author)

    snapshot = _analyze_clone(
        repository,
        str(tmp_path),
        head.hexsha,
        changed_paths=["included.py", "README.md"],
        base_sha=base.hexsha,
    )

    assert [item["file_path"] for item in snapshot.files] == ["included.py"]
    assert snapshot.commit_features["files_changed"] == 1
    assert snapshot.commit_features["lines_added"] == 1
