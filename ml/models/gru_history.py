"""GRU model for encoding per-file commit history sequences.

Architecture:
    Input:  (batch, N_commits_before=20, 7_history_features)
    Output: hidden state of shape (batch, hidden_dim=128)

history_features = [lines_added, lines_deleted, churn, complexity,
                    contributor_activity, commit_frequency, previous_defects]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HISTORY_FEATURES = [
    "lines_added",
    "lines_deleted",
    "churn",
    "complexity",
    "contributor_activity",
    "commit_frequency",
    "previous_defects",
]
N_HISTORY = 20
INPUT_SIZE = len(HISTORY_FEATURES)
HIDDEN_SIZE = 128


def build_gru(
    input_size: int = INPUT_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> Any:
    """Return an initialized GRU module (requires torch)."""
    import torch.nn as nn

    return nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )


def encode_history(gru: Any, sequences: Any) -> Any:
    """Run sequences (batch, seq_len, input_size) through GRU.

    Returns:
        FloatTensor of shape (batch, hidden_size) — last-layer hidden state.
    """
    _, hidden = gru(sequences)
    return hidden[-1]


def save_gru(gru: Any, config: dict, output_dir: Path) -> None:
    """Save GRU weights (.pt) and config (.json) to output_dir."""
    import io
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(gru.state_dict(), buf)
    (output_dir / "gru_weights.pt").write_bytes(buf.getvalue())
    (output_dir / "gru_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def load_gru(output_dir: Path) -> tuple[Any, dict]:
    """Load GRU weights and config from output_dir."""
    import torch

    config = json.loads(
        (output_dir / "gru_config.json").read_text(encoding="utf-8")
    )
    gru = build_gru(
        **{k: config[k] for k in ("input_size", "hidden_size", "num_layers", "dropout")}
    )
    gru.load_state_dict(
        torch.load(output_dir / "gru_weights.pt", map_location="cpu")
    )
    gru.eval()
    return gru, config
