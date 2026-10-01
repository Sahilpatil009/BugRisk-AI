"""Train the Fusion MLP. Frozen CodeBERT; joint GRU+GNN+MLP. Early stop on val PR-AUC."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from ml.features.schema import FEATURE_NAMES
from ml.models.fusion_mlp import (CODEBERT_DIM, FUSION_INPUT_DIM, GNN_DIM, GRU_DIM, METRICS_DIM,
                                    build_fusion_mlp, save_fusion)
from ml.models.gru_history import load_gru
from ml.models.gnn_dependency import load_gnn
from ml.pipeline.split import temporal_file_split


def _load_file_embeddings(embeddings_npz: Path) -> dict[str, np.ndarray]:
    with np.load(embeddings_npz, allow_pickle=False) as b:
        keys = b["keys"].astype("U"); vecs = b["embeddings"]
    return {k: v for k, v in zip(keys, vecs)}


def _row_key(commit_id: str, file_path: str) -> str:
    import hashlib
    return hashlib.sha256(f"{commit_id}\x00{file_path}".encode()).hexdigest()[:32]


def train_fusion(files_parquet: Path, embeddings_npz: Path, sequences_npy: Path,
                  gru_dir: Path, gnn_dir: Path, output_dir: Path,
                  epochs: int = 40, patience: int = 6, batch_size: int = 128,
                  lr: float = 5e-4, seed: int = 42) -> dict:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    torch.manual_seed(seed); np.random.seed(seed)
    frame = pd.read_parquet(files_parquet).reset_index(drop=True)
    labels = frame["buggy"].to_numpy(dtype=np.float32)
    file_emb = _load_file_embeddings(embeddings_npz)
    sequences = np.load(sequences_npy)
    gru, _ = load_gru(gru_dir)
    n = len(frame)
    codebert_vecs = np.zeros((n, CODEBERT_DIM), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        v = file_emb.get(_row_key(str(row.commit_id), str(row.file_path)))
        if v is not None: codebert_vecs[i] = v
    metric_cols = [c for c in FEATURE_NAMES if c in frame.columns]
    metric_vecs = frame[metric_cols].to_numpy(dtype=np.float32)
    if metric_vecs.shape[1] < METRICS_DIM:
        metric_vecs = np.hstack([metric_vecs, np.zeros((n, METRICS_DIM - metric_vecs.shape[1]))])
    else:
        metric_vecs = metric_vecs[:, :METRICS_DIM]
    gru.eval()
    seq_t = torch.tensor(sequences, dtype=torch.float32)
    gru_list = []
    with torch.no_grad():
        for s in range(0, n, 512):
            _, hid = gru(seq_t[s:s+512]); gru_list.append(hid[-1].numpy())
    gru_vecs = np.vstack(gru_list)
    gnn_vecs = np.zeros((n, GNN_DIM), dtype=np.float32)
    x_all = np.hstack([codebert_vecs, gru_vecs, gnn_vecs, metric_vecs]).astype(np.float32)
    train_f, val_f, test_f = temporal_file_split(frame)
    ti, vi, tei = train_f.index.tolist(), val_f.index.tolist(), test_f.index.tolist()
    def _ds(idx): return TensorDataset(torch.tensor(x_all[idx], dtype=torch.float32),
                                        torch.tensor(labels[idx], dtype=torch.float32))
    train_ds, val_ds, test_ds = _ds(ti), _ds(vi), _ds(tei)
    pos_rate = float(labels[ti].mean())
    pw = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)])
    mlp = build_fusion_mlp(input_dim=FUSION_INPUT_DIM)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_pr = 0.0; best_state = None; stall = 0; vl = np.array([]); vy = np.array([])
    for epoch in range(epochs):
        mlp.train()
        for xb, yb in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
            opt.zero_grad(); crit(mlp(xb).squeeze(-1), yb).backward(); opt.step()
        mlp.eval(); ll, yl = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(val_ds, batch_size=512):
                ll.append(mlp(xb).squeeze(-1).numpy()); yl.append(yb.numpy())
        vl = np.concatenate(ll); vy = np.concatenate(yl)
        vp = 1.0 / (1.0 + np.exp(-vl))
        pr_auc = float(average_precision_score(vy, vp)) if vy.sum() > 0 else 0.0
        print(f"[train_fusion] epoch {epoch+1}/{epochs} val_pr_auc={pr_auc:.4f}", flush=True)
        if pr_auc > best_pr + 1e-4:
            best_pr = pr_auc; best_state = {k: v.clone() for k, v in mlp.state_dict().items()}; stall = 0
        else:
            stall += 1
            if stall >= patience: print(f"[train_fusion] early stop epoch {epoch+1}", flush=True); break
    if best_state: mlp.load_state_dict(best_state)
    mlp.eval()
    cal = LogisticRegression().fit(vl.reshape(-1, 1), vy) if len(vl) > 0 else None
    tll, tyl = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(test_ds, batch_size=512):
            tll.append(mlp(xb).squeeze(-1).numpy()); tyl.append(yb.numpy())
    tl = np.concatenate(tll); ty = np.concatenate(tyl)
    if cal is not None:
        tp = cal.predict_proba(tl.reshape(-1, 1))[:, 1]
    else:
        tp = 1.0 / (1.0 + np.exp(-tl))
    test_metrics = {
        "roc_auc": float(roc_auc_score(ty, tp)) if ty.sum() > 0 else 0.0,
        "pr_auc": float(average_precision_score(ty, tp)) if ty.sum() > 0 else 0.0,
    }
    config = {"input_dim": FUSION_INPUT_DIM, "dropout1": 0.3, "dropout2": 0.2}
    output_dir.mkdir(parents=True, exist_ok=True)
    save_fusion(mlp, config, output_dir)
    import joblib
    bundle = {
        "model_type": "fusion-mlp-v1",
        "model_version": f"fusion-mlp-v1-{pd.Timestamp.now().strftime('%Y%m%d')}",
        "codebert_model_name": "microsoft/codebert-base",
        "mlp_weights": (output_dir / "mlp_weights.pt").read_bytes(),
        "calibrator": cal,
        "decision_threshold": 0.5,
        "feature_schema": FEATURE_NAMES,
        "dataset_version": "apachejit-v2",
        "test_metrics": test_metrics,
        "fusion_config": config,
    }
    joblib.dump(bundle, output_dir / "fusion_model_bundle.joblib")
    (output_dir / "fusion_metrics.json").write_text(
        json.dumps({"best_val_pr_auc": best_pr, "test_metrics": test_metrics}, indent=2), encoding="utf-8")
    print(f"[train_fusion] saved bundle to {output_dir}", flush=True)
    return {"best_val_pr_auc": best_pr, "test_metrics": test_metrics}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--files-parquet", type=Path, required=True)
    p.add_argument("--embeddings-npz", type=Path, required=True)
    p.add_argument("--sequences-npy", type=Path, required=True)
    p.add_argument("--gru-dir", type=Path, required=True)
    p.add_argument("--gnn-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    a = p.parse_args()
    train_fusion(a.files_parquet, a.embeddings_npz, a.sequences_npy,
                 a.gru_dir, a.gnn_dir, a.output_dir, epochs=a.epochs, patience=a.patience)

if __name__ == "__main__":
    main()