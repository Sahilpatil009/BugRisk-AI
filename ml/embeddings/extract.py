"""Extract the changed source code for ApacheJIT commits from GitHub mirrors.

The ApacheJIT dataset carries per-commit metrics but no source text.  For the
CodeBERT extension we need the actual code each commit changed, so this stage
clones the public Apache GitHub mirrors (into a cache directory that lives
outside the repository) and records the post-change content of every source
file touched by each dataset commit.

The extraction is done with two bulk git operations per mirror instead of one
process per commit:

1. ``git log --all --name-only`` maps every commit to its added/modified
   files in a single call.
2. ``git cat-file --batch`` streams the requested ``sha:path`` blobs in one
   process.

Output is a Parquet file of ``(project, commit_id, file_path, language,
content)`` rows plus a ``resolution.json`` report describing how many dataset
commits could actually be resolved in the mirror.
"""

import argparse
import io
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SOURCE_EXTENSIONS = {".java": "java", ".py": "python", ".scala": "scala", ".kt": "kotlin"}
# 0x1F (unit separator) suffix on each commit line: distinct from paths and,
# unlike a NUL byte, safe to pass through Windows subprocess argument lists.
COMMIT_SUFFIX = "\x1f"

# ApacheJIT names some entries after Maven modules that live in a shared
# repository on GitHub.
PROJECT_REPO_OVERRIDES = {
    "apache/hadoop-core": "apache/hadoop",
    "apache/hadoop-mapreduce": "apache/hadoop",
    "apache/hadoop-hdfs": "apache/hadoop",
    "apache/hadoop-yarn": "apache/hadoop",
}

DEFAULT_PROJECTS = [
    "apache/zookeeper",
    "apache/spark",
    "apache/hadoop-mapreduce",
    "apache/hadoop-hdfs",
    "apache/kafka",
]


@dataclass
class ExtractionResult:
    project: str
    repo: str
    total_commits: int = 0
    resolved_commits: int = 0
    commits_with_code: int = 0
    files_extracted: int = 0
    files_missing: int = 0
    details: dict = field(default_factory=dict)


def _run_git(repo_dir: Path, *args: str, timeout: int = 3600, stdin_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "-C", str(repo_dir), *args],
        capture_output=True,
        timeout=timeout,
        input=stdin_bytes,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message[:500]}")
    return result.stdout


def ensure_clone(repo: str, cache_root: Path) -> Path:
    """Clone ``https://github.com/{repo}.git`` into the cache, reusing existing clones."""
    destination = cache_root / repo.replace("/", "__")
    if (destination / ".git").is_dir():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[extract] cloning https://github.com/{repo}.git -> {destination}", flush=True)
    _run_git(
        destination.parent,
        "clone",
        "--quiet",
        f"https://github.com/{repo}.git",
        str(destination),
        timeout=7200,
    )
    return destination


def changed_files_by_commit(repo_dir: Path) -> dict[str, list[str]]:
    """Map every commit in all refs to its added/modified files (one git call)."""
    output = _run_git(
        repo_dir,
        "log",
        "--all",
        "--format=%H%x1f",
        "--name-only",
        "--diff-filter=AM",
        timeout=7200,
    ).decode("utf-8", errors="replace")
    files: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in output.splitlines():
        if line.endswith(COMMIT_SUFFIX):
            current = files.setdefault(line[: -len(COMMIT_SUFFIX)], [])
            continue
        if current is not None and line:
            current.append(line)
    return files


def _batch_blob_contents(repo_dir: Path, requests: list[str]) -> dict[str, str]:
    """Fetch ``sha:path`` blobs in a single ``git cat-file --batch`` process."""
    contents: dict[str, str] = {}
    if not requests:
        return contents
    stdin = ("\n".join(requests) + "\n").encode("utf-8")
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "cat-file", "--batch"],
        input=stdin,
        capture_output=True,
        timeout=7200,
        check=False,
    )
    stream = io.BytesIO(result.stdout)
    for request in requests:
        header = stream.readline().decode("utf-8", errors="replace").rstrip("\n")
        parts = header.split(" ")
        if len(parts) < 2 or parts[1] == "missing":
            continue  # missing objects carry no payload
        size = int(parts[2])
        raw = stream.read(size)
        stream.read(1)  # trailing newline
        if parts[1] == "blob":
            contents[request] = raw.decode("utf-8", errors="ignore")
    if result.returncode != 0:
        raise RuntimeError(f"git cat-file --batch failed: {result.stderr[:500]!r}")
    return contents


def extract_project(
    frame: pd.DataFrame,
    project: str,
    cache_root: Path,
    max_files: int = 8,
    max_chars: int = 20000,
) -> tuple[pd.DataFrame, ExtractionResult]:
    repo = PROJECT_REPO_OVERRIDES.get(project, project)
    repo_dir = ensure_clone(repo, cache_root)
    return extract_from_repo(frame, project, repo_dir, max_files=max_files, max_chars=max_chars)


def extract_from_repo(
    frame: pd.DataFrame,
    project: str,
    repo_dir: Path,
    max_files: int = 8,
    max_chars: int = 20000,
) -> tuple[pd.DataFrame, ExtractionResult]:
    repo = PROJECT_REPO_OVERRIDES.get(project, project)
    rows = frame[frame["project"] == project]
    wanted = [str(sha) for sha in rows["commit_id"].astype(str).unique()]
    result = ExtractionResult(project=project, repo=repo, total_commits=len(wanted))

    print(f"[extract] {project}: mapping changed files for {repo} ...", flush=True)
    files_by_commit = changed_files_by_commit(repo_dir)
    result.resolved_commits = sum(1 for sha in wanted if sha in files_by_commit)

    records: list[dict[str, str]] = []
    requests: list[str] = []
    commit_of_request: list[str] = []
    for sha in wanted:
        paths = files_by_commit.get(sha)
        if not paths:
            continue
        source_paths = [
            path
            for path in paths
            if Path(path).suffix in SOURCE_EXTENSIONS
        ][:max_files]
        if not source_paths:
            continue
        result.commits_with_code += 1
        for path in source_paths:
            request = f"{sha}:{path}"
            requests.append(request)
            commit_of_request.append(sha)
    print(f"[extract] {project}: reading {len(requests)} file contents ...", flush=True)
    contents = _batch_blob_contents(repo_dir, requests)
    for request, sha in zip(requests, commit_of_request, strict=True):
        content = contents.get(request)
        if content is None:
            result.files_missing += 1
            continue
        path = request.split(":", 1)[1]
        records.append(
            {
                "project": project,
                "commit_id": sha,
                "file_path": path,
                "language": SOURCE_EXTENSIONS[Path(path).suffix],
                "content": content[:max_chars],
            }
        )
    result.files_extracted = len(records)
    source_frame = pd.DataFrame(records, columns=["project", "commit_id", "file_path", "language", "content"])
    print(
        f"[extract] {project}: resolved {result.resolved_commits}/{result.total_commits} commits, "
        f"{result.commits_with_code} with source, {result.files_extracted} files",
        flush=True,
    )
    return source_frame, result


def extract_sources(
    frame: pd.DataFrame,
    projects: list[str],
    cache_root: Path,
    output_root: Path,
    max_files: int = 8,
    max_chars: int = 20000,
) -> tuple[pd.DataFrame, dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    report: dict = {}
    for project in projects:
        if project not in set(frame["project"]):
            raise ValueError(f"Project {project} is not present in the dataset")
        source_frame, result = extract_project(
            frame, project, cache_root, max_files=max_files, max_chars=max_chars
        )
        frames.append(source_frame)
        report[project] = {
            "repo": result.repo,
            "total_commits": result.total_commits,
            "resolved_commits": result.resolved_commits,
            "commits_with_code": result.commits_with_code,
            "files_extracted": result.files_extracted,
            "files_missing": result.files_missing,
        }
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(output_root / "sources.parquet", index=False)
    (output_root / "resolution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return combined, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Processed ApacheJIT CSV with commit_id")
    parser.add_argument("--projects", type=str, default=",".join(DEFAULT_PROJECTS))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "bugrisk" / "repos",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=20000)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    if "commit_id" not in frame.columns:
        raise ValueError("Processed dataset must include commit_id (re-run the preprocess stage)")
    extract_sources(
        frame,
        [project.strip() for project in args.projects.split(",") if project.strip()],
        args.cache_root,
        args.output,
        max_files=args.max_files,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
