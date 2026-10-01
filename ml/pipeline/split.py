"""Chronological per-project split for file-level rows.

Ensures no file from a future commit leaks into training by splitting
on commit boundaries (not individual file rows).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def temporal_file_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames with 60/20/20 chronological split."""
    train_parts, val_parts, test_parts = [], [], []
    for _, project in frame.sort_values("event_time").groupby("project", sort=False):
        unique_commits = project.drop_duplicates("commit_id").sort_values("event_time")
        n = len(unique_commits)
        train_end = max(1, int(n * 0.6))
        val_end = max(train_end + 1, int(n * 0.8))
        train_commits = set(unique_commits.iloc[:train_end]["commit_id"])
        val_commits = set(unique_commits.iloc[train_end:val_end]["commit_id"])
        train_parts.append(project[project["commit_id"].isin(train_commits)])
        val_parts.append(project[project["commit_id"].isin(val_commits)])
        test_parts.append(project[~project["commit_id"].isin(train_commits | val_commits)])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal file-level split")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    train, val, test = temporal_file_split(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        out = args.output_dir / f"{name}.parquet"
        part.to_parquet(out, index=False)
        print(f"[split] {name}: {len(part)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
