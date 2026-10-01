"""Fusion MLP combining CodeBERT (768) + GRU (128) + GNN (64) + metrics (11) = 971-d."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

CODEBERT_DIM = 768
GRU_DIM = 128
GNN_DIM = 64
METRICS_DIM = 11
FUSION_INPUT_DIM = CODEBERT_DIM + GRU_DIM + GNN_DIM + METRICS_DIM  # 971


def build_fusion_mlp(input_dim: int = FUSION_INPUT_DIM,
                      dropout1: float = 0.3, dropout2: float = 0.2) -> Any:
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout1),
        nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout2),
        nn.Linear(128, 1),
    )


def save_fusion(mlp: Any, config: dict, output_dir: Path) -> None:
    import io, torch
    output_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(mlp.state_dict(), buf)
    (output_dir / "mlp_weights.pt").write_bytes(buf.getvalue())
    (output_dir / "fusion_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_fusion(output_dir: Path) -> tuple[Any, dict]:
    import torch
    config = json.loads((output_dir / "fusion_config.json").read_text(encoding="utf-8"))
    mlp = build_fusion_mlp(input_dim=config["input_dim"],
                            dropout1=config["dropout1"], dropout2=config["dropout2"])
    mlp.load_state_dict(torch.load(output_dir / "mlp_weights.pt", map_location="cpu"))
    mlp.eval()
    return mlp, config
