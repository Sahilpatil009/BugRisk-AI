import ast
import base64
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from git import Repo
from radon.complexity import cc_visit  # type: ignore[import-untyped]


@dataclass
class RepositorySnapshot:
    commit_sha: str
    commit_features: dict[str, float]
    files: list[dict[str, float | int | str]]


def _dependencies(source: str) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return len(imports)


def _entropy(values: list[int]) -> float:
    total = sum(values)
    if not total:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in values if value)


def _clone_environment(token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if token:
        credential = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {credential}",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
    return env


def analyze_repository(
    url: str, branch: str, token: str | None = None, commit_sha: str | None = None
) -> RepositorySnapshot:
    with tempfile.TemporaryDirectory(prefix="bugrisk-") as directory:
        with Repo.clone_from(
            url,
            directory,
            branch=branch,
            env=_clone_environment(token),
            multi_options=["--depth=100"],
        ) as repo:
            return _analyze_clone(repo, directory, commit_sha)


def _analyze_clone(repo: Repo, directory: str, commit_sha: str | None) -> RepositorySnapshot:
    if commit_sha:
        repo.git.checkout(commit_sha)
    head = repo.head.commit
    python_paths = [path for path in Path(directory).rglob("*.py") if ".git" not in path.parts]
    if not python_paths:
        raise ValueError("No Python files were found in the selected repository")

    commits = list(repo.iter_commits(max_count=100))
    author_counts = Counter(str(commit.author.email or commit.author.name) for commit in commits)
    head_author = str(head.author.email or head.author.name)
    stats = head.stats.files
    changes = [int(value.get("lines", 0)) for value in stats.values()]
    added = sum(int(value.get("insertions", 0)) for value in stats.values())
    deleted = sum(int(value.get("deletions", 0)) for value in stats.values())
    now = datetime.now(UTC)
    commit_features = {
        "lines_added": float(added),
        "lines_deleted": float(deleted),
        "files_changed": float(max(len(stats), 1)),
        "code_churn": float(added + deleted),
        "commit_entropy": _entropy(changes),
        "developer_experience": float(author_counts[head_author]),
        "file_age_days": 0.0,
        "commit_frequency": float(
            sum(1 for commit in commits if (now - commit.committed_datetime).days <= 30)
        ),
        "contributor_count": float(len(author_counts)),
        "complexity": 0.0,
        "previous_defects": float(
            sum(
                1
                for commit in commits
                if any(word in commit.message.lower() for word in ("fix", "bug", "defect"))
            )
        ),
    }

    file_results: list[dict[str, float | int | str]] = []
    complexities: list[float] = []
    ages: list[int] = []
    for path in python_paths[:500]:
        relative = path.relative_to(directory).as_posix()
        source = path.read_text(encoding="utf-8", errors="ignore")
        history = list(repo.iter_commits(paths=relative, max_count=100))
        contributors = {str(commit.author.email or commit.author.name) for commit in history}
        first_date = history[-1].committed_datetime if history else head.committed_datetime
        age_days = max((now - first_date).days, 0)
        complexity = float(sum(block.complexity for block in cc_visit(source)))
        file_stats: Any = stats.get(relative, {})
        lines_added = int(file_stats.get("insertions", 0))
        lines_deleted = int(file_stats.get("deletions", 0))
        previous_defects = sum(
            1
            for commit in history
            if any(word in commit.message.lower() for word in ("fix", "bug", "defect"))
        )
        item: dict[str, float | int | str] = {
            "file_path": relative,
            "loc": sum(1 for line in source.splitlines() if line.strip()),
            "complexity": round(complexity, 2),
            "code_churn": lines_added + lines_deleted,
            "commit_count": len(history),
            "contributor_count": len(contributors),
            "file_age_days": age_days,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "dependency_count": _dependencies(source),
            "previous_defects": previous_defects,
        }
        file_results.append(item)
        complexities.append(complexity)
        ages.append(age_days)

    commit_features["complexity"] = float(sum(complexities) / max(len(complexities), 1))
    commit_features["file_age_days"] = float(sum(ages) / max(len(ages), 1))
    return RepositorySnapshot(head.hexsha, commit_features, file_results)
