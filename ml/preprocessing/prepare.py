import argparse
from pathlib import Path

import pandas as pd

from ml.features.schema import (
    FEATURE_ALIASES,
    FEATURE_NAMES,
    PROJECT_ALIASES,
    TARGET_ALIASES,
    TIME_ALIASES,
)


def _first(columns: set[str], names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in columns), None)


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    columns = set(frame.columns)
    target = _first(columns, TARGET_ALIASES)
    if not target:
        raise ValueError("Dataset must contain the ApacheJIT 'buggy' target")

    prepared = pd.DataFrame(index=frame.index)
    for canonical, aliases in FEATURE_ALIASES.items():
        source = _first(columns, aliases)
        prepared[canonical] = (
            pd.to_numeric(frame[source], errors="coerce") if source else 0.0
        )
    prepared["code_churn"] = prepared["lines_added"].fillna(0) + prepared[
        "lines_deleted"
    ].fillna(0)
    prepared["buggy"] = (
        frame[target].astype(str).str.lower().isin({"1", "true", "yes"}).astype(int)
    )
    time_column = _first(columns, TIME_ALIASES)
    project_column = _first(columns, PROJECT_ALIASES)
    prepared["event_time"] = _parse_event_time(frame, time_column)
    prepared["project"] = (
        frame[project_column].astype(str) if project_column else "apachejit"
    )
    prepared["commit_id"] = (
        frame[_first(columns, ("commit_id", "commit"))].astype(str)
        if _first(columns, ("commit_id", "commit"))
        else ""
    )
    prepared = (
        prepared.dropna(subset=["event_time"])
        .sort_values(["project", "event_time"])
        .reset_index(drop=True)
    )
    if prepared.empty or prepared["buggy"].nunique() < 2:
        raise ValueError("Prepared dataset must contain both buggy and clean examples")
    return prepared[[*FEATURE_NAMES, "buggy", "event_time", "project", "commit_id"]]


def _parse_event_time(frame: pd.DataFrame, time_column: str | None) -> pd.Series:
    """Parse epoch timestamps (ApacheJIT author_date) or date strings into tz-aware times."""
    if not time_column:
        return pd.date_range("2000-01-01", periods=len(frame), freq="s", tz="UTC")
    numeric = pd.to_numeric(frame[time_column], errors="coerce")
    if numeric.notna().any():
        # Unix seconds for this era of data are ~1e9; values above 1e12 are milliseconds.
        unit = "ms" if float(numeric.dropna().max()) > 1e12 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(frame[time_column], errors="coerce", utc=True)


def _check_no_user_repos(sample: pd.DataFrame) -> None:
    """Raise if the dataset contains non-Apache GitHub repository URLs."""
    if "source_url" not in sample.columns:
        return  # No source_url column means it's the standard ApacheJIT format.
    import re
    non_apache = sample["source_url"].dropna().apply(
        lambda url: bool(re.search(r"github\.com/(?!apache/)", str(url)))
    )
    if non_apache.any():
        raise ValueError(
            "Dataset contains non-Apache GitHub repository URLs. "
            "Pass --allow-user-repos to allow user repositories (requires explicit consent)."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-user-repos", action="store_true", default=False,
                        help="Allow non-Apache GitHub repos in the dataset (requires explicit consent).")
    args = parser.parse_args()

    # Reject datasets that appear to contain live user repositories unless the
    # operator explicitly opts in with --allow-user-repos.
    if not args.allow_user_repos:
        _check_no_user_repos(pd.read_csv(args.input, nrows=5, usecols=lambda c: c == "source_url", on_bad_lines="skip"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepare_frame(pd.read_csv(args.input)).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
