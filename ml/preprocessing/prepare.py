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
    prepared["event_time"] = (
        pd.to_datetime(frame[time_column], errors="coerce", utc=True)
        if time_column
        else pd.date_range("2000-01-01", periods=len(frame), freq="s", tz="UTC")
    )
    prepared["project"] = (
        frame[project_column].astype(str) if project_column else "apachejit"
    )
    prepared = (
        prepared.dropna(subset=["event_time"])
        .sort_values(["project", "event_time"])
        .reset_index(drop=True)
    )
    if prepared.empty or prepared["buggy"].nunique() < 2:
        raise ValueError("Prepared dataset must contain both buggy and clean examples")
    return prepared[[*FEATURE_NAMES, "buggy", "event_time", "project"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepare_frame(pd.read_csv(args.input)).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
