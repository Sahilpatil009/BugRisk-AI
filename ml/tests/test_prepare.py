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
        }
    )
    prepared = prepare_frame(raw)
    assert list(prepared.columns) == [*FEATURE_NAMES, "buggy", "event_time", "project"]
    assert prepared.iloc[1]["code_churn"] == 14
