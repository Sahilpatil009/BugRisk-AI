"""Build fixed-length padded history sequences for the GRU model."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ml.models.gru_history import HISTORY_FEATURES, N_HISTORY


def _build_sequences(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = frame.reset_index(drop=True).sort_values(["project", "file_path", "event_time"])
    n = len(frame)
    sequences = np.zeros((n, N_HISTORY, len(HISTORY_FEATURES)), dtype=np.float32)
    masks = np.zeros((n, N_HISTORY), dtype=bool)
    present = [f for f in HISTORY_FEATURES if f in frame.columns]
    for (_, _), group in frame.groupby(["project", "file_path"], sort=False):
        group = group.sort_values("event_time")
        for pos, row_idx in enumerate(group.index.tolist()):
            history = group.iloc[max(0, pos - N_HISTORY): pos]
            if history.empty:
                continue
            hist_len = len(history)
            feat_vals = history[present].to_numpy(dtype=np.float32)
            pad_start = N_HISTORY - hist_len
            sequences[row_idx, pad_start:, : len(present)] = feat_vals
            masks[row_idx, pad_start:] = True
    return sequences, masks


def build_history_sequences(files_parquet: Path, output_dir: Path) -> None:
    frame = pd.read_parquet(files_parquet).reset_index(drop=True)
    sequences, masks = _build_sequences(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "history_sequences.npy", sequences)
    np.save(output_dir / "history_masks.npy", masks)
    np.savez_compressed(output_dir / "history_keys.npz",
                        commit_id=frame["commit_id"].to_numpy().astype("U"),
                        file_path=frame["file_path"].to_numpy().astype("U"))
    print(f"[history_sequences] built {len(sequences)} sequences", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--files-parquet", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    build_history_sequences(a.files_parquet, a.output_dir)

if __name__ == "__main__":
    main()
