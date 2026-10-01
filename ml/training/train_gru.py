"""Train the GRU history model. Weighted BCE, early stopping, Platt calibration."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from ml.models.gru_history import HISTORY_FEATURES, N_HISTORY, build_gru, save_gru
from ml.pipeline.split import temporal_file_split


def train_gru(files_parquet: Path, sequences_dir: Path, output_dir: Path,
              hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2,
              epochs: int = 30, patience: int = 5, batch_size: int = 256,
              lr: float = 1e-3, seed: int = 42) -> dict:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    frame = pd.read_parquet(files_parquet).reset_index(drop=True)
    sequences = np.load(sequences_dir / "history_sequences.npy")
    labels = frame["buggy"].to_numpy(dtype=np.float32)
    train_frame, val_frame, _ = temporal_file_split(frame)
    train_idx = train_frame.index.tolist()
    val_idx = val_frame.index.tolist()
    def _ds(idx): return TensorDataset(torch.tensor(sequences[idx], dtype=torch.float32),
                                        torch.tensor(labels[idx], dtype=torch.float32))
    train_ds, val_ds = _ds(train_idx), _ds(val_idx)
    pos_rate = float(labels[train_idx].mean())
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)])
    gru = build_gru(input_size=len(HISTORY_FEATURES), hidden_size=hidden_size,
                    num_layers=num_layers, dropout=dropout)
    head = nn.Linear(hidden_size, 1)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val = float("inf"); best_g = None; best_h = None; stall = 0
    for epoch in range(epochs):
        gru.train(); head.train()
        for sb, yb in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
            opt.zero_grad(); _, hid = gru(sb); crit(head(hid[-1]).squeeze(-1), yb).backward(); opt.step()
        gru.eval(); head.eval(); vl = []
        with torch.no_grad():
            for sb, yb in DataLoader(val_ds, batch_size=512):
                _, hid = gru(sb); vl.append(crit(head(hid[-1]).squeeze(-1), yb).item())
        vl = float(np.mean(vl))
        print(f"[train_gru] epoch {epoch+1}/{epochs} val_loss={vl:.4f}", flush=True)
        if vl < best_val - 1e-4:
            best_val = vl; best_g = {k: v.clone() for k, v in gru.state_dict().items()}
            best_h = {k: v.clone() for k, v in head.state_dict().items()}; stall = 0
        else:
            stall += 1
            if stall >= patience: print(f"[train_gru] early stop epoch {epoch+1}", flush=True); break
    if best_g: gru.load_state_dict(best_g)
    if best_h: head.load_state_dict(best_h)
    from sklearn.linear_model import LogisticRegression
    gru.eval(); head.eval(); ll, yl = [], []
    with torch.no_grad():
        for sb, yb in DataLoader(val_ds, batch_size=512):
            _, hid = gru(sb); ll.append(head(hid[-1]).squeeze(-1).numpy()); yl.append(yb.numpy())
    cal = LogisticRegression().fit(np.concatenate(ll).reshape(-1, 1), np.concatenate(yl))
    config = {"input_size": len(HISTORY_FEATURES), "hidden_size": hidden_size,
               "num_layers": num_layers, "dropout": dropout,
               "history_features": HISTORY_FEATURES, "n_history": N_HISTORY}
    output_dir.mkdir(parents=True, exist_ok=True)
    save_gru(gru, config, output_dir)
    import joblib
    joblib.dump({"head": head.state_dict(), "calibrator": cal}, output_dir / "gru_head.joblib")
    metrics = {"best_val_loss": best_val}
    (output_dir / "gru_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train_gru] saved to {output_dir}", flush=True)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--files-parquet", type=Path, required=True)
    p.add_argument("--sequences-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--hidden-size", type=int, default=128)
    a = p.parse_args()
    train_gru(a.files_parquet, a.sequences_dir, a.output_dir,
              hidden_size=a.hidden_size, epochs=a.epochs, patience=a.patience)

if __name__ == "__main__":
    main()
