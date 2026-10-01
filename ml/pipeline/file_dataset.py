"""Build a per-file dataset row from the ApacheJIT commit-level CSV.

For each (project, commit_id) row in the processed ApacheJIT dataset this
module emits one row per changed source file, enriching it with per-file
metrics from the commit row and a chronological split label.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _chronological_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a 'split' column using 60/20/20 chronological per-project split."""
    parts: list[pd.DataFrame] = []
    for _, project in frame.sort_values("event_time").groupby("project", sort=False):
        n = len(project)
        train_end = max(1, int(n * 0.6))
        val_end = max(train_end + 1, int(n * 0.8))
        proj = project.copy()
        proj["split"] = "test"
        proj.iloc[:train_end, proj.columns.get_loc("split")] = "train"
        proj.iloc[train_end:val_end, proj.columns.get_loc("split")] = "val"
        parts.append(proj)
    return pd.concat(parts, ignore_index=True)


def build_file_dataset(
    processed_csv: Path,
    sources_parquet: Path,
    output_parquet: Path,
) -> pd.DataFrame:
    """Join commit-level rows with per-file source content and compute splits."""
    commits = pd.read_csv(processed_csv, parse_dates=["event_time"])
    sources = pd.read_parquet(sources_parquet)

    commit_cols = [
        "project", "commit_id", "buggy", "event_time",
        "lines_added", "lines_deleted", "code_churn", "complexity",
        "commit_frequency", "contributor_count", "file_age_days",
        "previous_defects",
    ]
    available = [c for c in commit_cols if c in commits.columns]
    merged = sources.merge(commits[available], on=["project", "commit_id"], how="inner")

    if "dependency_count" not in merged.columns:
        merged["dependency_count"] = 0
    if "commit_count" not in merged.columns:
        merged["commit_count"] = merged.get("commit_frequency", pd.Series(0, index=merged.index))

    merged["source_hash"] = merged["content"].apply(_content_hash)
    merged = _chronological_split(merged)

    output_cols = [
        "project", "commit_id", "file_path", "event_time", "buggy",
        "lines_added", "lines_deleted", "complexity", "code_churn",
        "commit_count", "contributor_count", "file_age_days",
        "dependency_count", "previous_defects", "content", "source_hash", "split",
    ]
    out = merged[[c for c in output_cols if c in merged.columns]].copy()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, index=False)
    print(
        f"[file_dataset] wrote {len(out)} rows "
        f"({int(out['buggy'].sum())} buggy) to {output_parquet}",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-file dataset")
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--sources-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_file_dataset(args.processed_csv, args.sources_parquet, args.output)


if __name__ == "__main__":
    main()
