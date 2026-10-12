import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.advanced.prepare import (
    build_graph_archive,
    build_temporal_histories,
    dependency_adjacency,
    prepare_advanced_inputs,
)
from ml.features.schema import FEATURE_NAMES


def _metric_frame(rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.gamma(2.0, 10.0, size=(rows, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    frame["buggy"] = (rng.random(rows) < 0.3).astype(int)
    frame["project"] = "demo"
    frame["commit_id"] = [f"{index:040x}" for index in range(rows)]
    frame["event_time"] = pd.date_range("2015-01-01", periods=rows, freq="D", tz="UTC")
    return frame


def test_histories_only_contain_strictly_earlier_commits():
    frame = _metric_frame(6)
    frame["lines_added"] = np.arange(6, dtype=float) * 100
    # Commits 3 and 4 share a timestamp: neither may see the other.
    frame.loc[4, "event_time"] = frame.loc[3, "event_time"]
    shuffled = frame.sample(frac=1.0, random_state=1)
    histories, masks, commit_ids, _ = build_temporal_histories(shuffled, window=3)
    by_commit = {commit: index for index, commit in enumerate(commit_ids)}
    column = FEATURE_NAMES.index("lines_added")

    first = by_commit[frame.loc[0, "commit_id"]]
    assert not masks[first].any()
    third = by_commit[frame.loc[2, "commit_id"]]
    assert masks[third].tolist() == [False, True, True]
    assert histories[third, masks[third], column].tolist() == [0.0, 100.0]
    for tied in (3, 4):
        row = by_commit[frame.loc[tied, "commit_id"]]
        assert histories[row, masks[row], column].tolist() == [0.0, 100.0, 200.0]
    last = by_commit[frame.loc[5, "commit_id"]]
    assert histories[last, :, column].tolist() == [200.0, 300.0, 400.0]


def test_dependency_edges_for_java_scala_and_python():
    files = pd.DataFrame(
        {
            "file_path": [
                "core/src/main/java/org/apache/zk/Server.java",
                "core/src/main/java/org/apache/zk/util/Clock.java",
                "core/src/main/scala/org/apache/zk/Plan.scala",
                "tools/pkg/report.py",
                "tools/pkg/render.py",
            ],
            "content": [
                "package org.apache.zk;\nimport org.apache.zk.util.Clock;\nimport java.util.List;\n",
                "package org.apache.zk.util;\nimport static org.apache.zk.Server.start;\n",
                "import org.apache.zk.{Server, util}\n",
                "from pkg.render import draw\n",
                "import os\n",
            ],
        }
    )
    _, adjacency, paths = dependency_adjacency(files)
    index = {path.rsplit("/", 1)[-1]: position for position, path in enumerate(paths)}
    edges = {
        (source, target)
        for source in index
        for target in index
        if adjacency[index[source], index[target]]
    }
    assert edges == {
        ("Server.java", "Clock.java"),
        ("Clock.java", "Server.java"),
        ("Plan.scala", "Server.java"),
        ("report.py", "render.py"),
    }
    assert np.diag(adjacency).sum() == 0


def test_graph_archive_pads_and_caps_nodes(tmp_path: Path):
    sources = pd.DataFrame(
        {
            "project": ["demo"] * 4,
            "commit_id": ["a1", "a1", "a1", "b2"],
            "file_path": ["A.py", "B.py", "C.py", "A.py"],
            "content": ["import B\n", "x = 1\n", "y = 2\n", "z = 3\n"],
        }
    )
    meta = build_graph_archive(sources, tmp_path / "graphs.npz", max_nodes=2)
    assert meta["graphs"] == 2
    with np.load(tmp_path / "graphs.npz", allow_pickle=False) as bundle:
        assert bundle["commit_id"].tolist() == ["a1", "b2"]
        assert bundle["node_features"].shape == (2, 2, 5)
        assert bundle["node_mask"].tolist() == [[True, True], [True, False]]
        assert bundle["adjacency"][0, 0, 1] == 1.0


@pytest.mark.parametrize(
    "switches",
    [
        {"use_codebert": False, "use_temporal": False, "use_graph": False},
        {"use_codebert": True, "use_temporal": False, "use_graph": False},
        {"use_codebert": True, "use_temporal": True, "use_graph": False},
        {"use_codebert": True, "use_temporal": True, "use_graph": True},
    ],
)
def test_fusion_model_shapes_and_probabilities(switches):
    torch = pytest.importorskip("torch")
    from ml.training.advanced_model import FusionConfig, build_fusion_model

    config = FusionConfig(metric_dim=11, codebert_dim=8, hidden_dim=16, **switches)
    model = build_fusion_model(config)
    batch = 5
    history_mask = torch.zeros(batch, 4, dtype=torch.bool)
    history_mask[1:, -2:] = True
    graph_mask = torch.zeros(batch, 3, dtype=torch.bool)
    graph_mask[:, 0] = True
    inputs = (
        torch.randn(batch, 11),
        torch.randn(batch, 8),
        torch.randn(batch, 4, 11),
        history_mask,
        torch.randn(batch, 3, 5),
        torch.ones(batch, 3, 3),
        graph_mask,
    )
    logits = model(*inputs)
    assert logits.shape == (batch,)
    probability = model.predict_probability(*inputs)
    assert bool(((probability >= 0) & (probability <= 1)).all())
    expected_inputs = 11 + 8 * switches["use_codebert"] + 32 * switches["use_temporal"]
    expected_inputs += 32 * switches["use_graph"]
    assert model.classifier[0].in_features == expected_inputs
    assert (model.gru is not None) == switches["use_temporal"]
    assert (model.graph is not None) == switches["use_graph"]


def test_train_advanced_experiment_end_to_end(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    from ml.training.train_advanced import train_advanced_experiment

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    rows = 300
    frame = _metric_frame(rows)
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    rng = np.random.default_rng(3)
    embedding = rng.normal(0, 1, size=(rows, 8)).astype(np.float32)
    embedding[frame["buggy"].to_numpy() == 1, :2] += 2.0
    embeddings_path = tmp_path / "embeddings.npz"
    np.savez(embeddings_path, commit_id=frame["commit_id"].to_numpy(dtype="S40"), embeddings=embedding)
    sources = pd.DataFrame(
        {
            "project": "demo",
            "commit_id": np.repeat(frame["commit_id"].to_numpy(), 2),
            "file_path": ["pkg/a.py", "pkg/b.py"] * rows,
            "content": ["from pkg.b import x\n", "x = 1\n"] * rows,
        }
    )
    sources_path = tmp_path / "sources.parquet"
    sources.to_parquet(sources_path, index=False)
    inputs = tmp_path / "advanced"
    meta = prepare_advanced_inputs(data_path, sources_path, inputs, window=3, max_nodes=4)
    assert meta["graphs"] == rows and meta["graphs_with_edges"] == rows

    result = train_advanced_experiment(
        data_path,
        embeddings_path,
        inputs / "temporal_histories.npz",
        inputs / "dependency_graphs.npz",
        tmp_path / "artifacts",
        hidden_dim=16,
        temporal_dim=8,
        graph_dim=8,
        epochs=3,
        patience=2,
        seeds=(1, 2),
        device="cpu",
    )
    assert set(result["variants"]) == {
        "xgboost_metrics",
        "mlp_metrics",
        "mlp_metrics_codebert",
        "mlp_metrics_codebert_gru",
        "mlp_metrics_codebert_gru_gnn",
    }
    assert result["subset"]["commits"] == rows
    for block in result["variants"].values():
        assert len(block["runs"]) == 2
        assert 0.0 <= block["summary"]["pr_auc"]["mean"] <= 1.0
        assert "confusion_matrix" in block["runs"][0]["test"]
    assert "mlp_metrics_codebert_gru_gnn_vs_mlp_metrics_codebert_gru" in result["deltas"]
    saved = json.loads((tmp_path / "artifacts" / "ablation.json").read_text(encoding="utf-8"))
    assert saved["best"]["variant"] in result["variants"]
    assert (tmp_path / "artifacts" / "advanced_model.joblib").exists()
