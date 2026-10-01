import pandas as pd

from ml.features.schema import FEATURE_NAMES
from ml.preprocessing.prepare import prepare_frame


def test_prepare_maps_apachejit_short_columns():
    raw = pd.DataFrame(
        {
            "la": [10, 3],
            "ld": [4, 1],
            "nf": [2, 1],
            "entropy": [0.5, 0.1],
            "exp": [5, 8],
            "age": [20, 40],
            "nuc": [3, 2],
            "ndev": [2, 1],
            "buggy": [1, 0],
            "date": ["2024-01-02", "2024-01-01"],
            "project": ["demo", "demo"],
            "commit_id": ["aaa111", "bbb222"],
        }
    )
    prepared = prepare_frame(raw)
    assert list(prepared.columns) == [*FEATURE_NAMES, "buggy", "event_time", "project", "commit_id"]
    assert prepared.iloc[1]["code_churn"] == 14
    # prepare sorts by event_time, so the 2024-01-01 row comes first.
    assert list(prepared["commit_id"]) == ["bbb222", "aaa111"]


def test_prepare_parses_epoch_seconds_not_nanoseconds():
    raw = pd.DataFrame(
        {
            "la": [10, 3],
            "ld": [4, 1],
            "nf": [2, 1],
            "entropy": [0.5, 0.1],
            "exp": [5, 8],
            "age": [20, 40],
            "nuc": [3, 2],
            "ndev": [2, 1],
            "buggy": [1, 0],
            "author_date": [1070355653, 1262304000],
            "project": ["demo", "demo"],
        }
    )
    prepared = prepare_frame(raw)
    assert prepared["event_time"].iloc[0] == pd.Timestamp("2003-12-02 09:00:53", tz="UTC")
    assert prepared["event_time"].iloc[1] == pd.Timestamp("2010-01-01 00:00:00", tz="UTC")
    assert (prepared["event_time"].iloc[0] < prepared["event_time"].iloc[1])
