"""GraphSAGE model for encoding file-level dependency graphs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

GNN_OUTPUT_DIM = 64
NODE_FEATURE_DIM = 11


def build_graphsage(in_channels: int = NODE_FEATURE_DIM, hidden_channels: int = 128,
                     out_channels: int = GNN_OUTPUT_DIM, num_layers: int = 2,
                     dropout: float = 0.2) -> Any:
    from torch_geometric.nn import GraphSAGE
    return GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels,
                     num_layers=num_layers, out_channels=out_channels,
                     dropout=dropout, aggr="mean")


def save_gnn(gnn: Any, config: dict, output_dir: Path) -> None:
    import io, torch
    output_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(gnn.state_dict(), buf)
    (output_dir / "gnn_weights.pt").write_bytes(buf.getvalue())
    (output_dir / "gnn_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_gnn(output_dir: Path) -> tuple[Any, dict]:
    import torch
    config = json.loads((output_dir / "gnn_config.json").read_text(encoding="utf-8"))
    gnn = build_graphsage(in_channels=config["in_channels"], hidden_channels=config["hidden_channels"],
                           out_channels=config["out_channels"], num_layers=config["num_layers"],
                           dropout=config["dropout"])
    gnn.load_state_dict(torch.load(output_dir / "gnn_weights.pt", map_location="cpu"))
    gnn.eval()
    return gnn, config
