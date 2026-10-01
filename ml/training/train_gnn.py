"""Train GraphSAGE on per-commit dependency graphs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from ml.models.gnn_dependency import GNN_OUTPUT_DIM, NODE_FEATURE_DIM, build_graphsage, save_gnn


def train_gnn(files_parquet: Path, graphs_dir: Path, output_dir: Path,
              hidden_channels: int = 128, num_layers: int = 2, dropout: float = 0.2,
              epochs: int = 20, patience: int = 4, lr: float = 1e-3, seed: int = 42) -> dict:
    import torch, torch.nn as nn
    try:
        import torch_geometric.data as pyg_data
    except ImportError as exc:
        raise ImportError("torch-geometric required. pip install torch-geometric") from exc
    torch.manual_seed(seed); np.random.seed(seed)
    frame = pd.read_parquet(files_parquet)
    meta = json.loads((graphs_dir / "graph_meta.json").read_text(encoding="utf-8"))
    commit_to_file = {e["commit_id"]: e["file"] for e in meta}
    gnn = build_graphsage(in_channels=NODE_FEATURE_DIM, hidden_channels=hidden_channels,
                           out_channels=GNN_OUTPUT_DIM, num_layers=num_layers, dropout=dropout)
    head = nn.Linear(GNN_OUTPUT_DIM, 1)
    opt = torch.optim.Adam(list(gnn.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    commit_ids = list(commit_to_file.keys())
    n = len(commit_ids)
    train_c = set(commit_ids[:int(n * 0.6)])
    val_c = set(commit_ids[int(n * 0.6):int(n * 0.8)])

    def _load(cid: str):
        fname = commit_to_file.get(cid)
        if not fname: return None
        npz = np.load(graphs_dir / fname, allow_pickle=False)
        node_ids = npz["node_ids"].astype("U")
        ei = torch.tensor(npz["edge_index"], dtype=torch.long)
        nf = torch.tensor(npz["node_features"], dtype=torch.float32)
        nf = (torch.cat([nf, torch.zeros(nf.shape[0], NODE_FEATURE_DIM - nf.shape[1])], dim=1)
              if nf.shape[1] < NODE_FEATURE_DIM else nf[:, :NODE_FEATURE_DIM])
        cr = frame[frame["commit_id"] == cid].reset_index(drop=True)
        nm = {nid: i for i, nid in enumerate(node_ids)}
        y = torch.zeros(len(node_ids), dtype=torch.float32)
        for _, row in cr.iterrows():
            idx = nm.get(str(row["file_path"]))
            if idx is not None: y[idx] = float(row["buggy"])
        return pyg_data.Data(x=nf, edge_index=ei, y=y)

    best_val = float("inf"); best_g = None; best_h = None; stall = 0
    for epoch in range(epochs):
        gnn.train(); head.train()
        for cid in train_c:
            d = _load(cid)
            if d is None or d.num_nodes == 0: continue
            opt.zero_grad(); emb = gnn(d.x, d.edge_index); crit(head(emb).squeeze(-1), d.y).backward(); opt.step()
        gnn.eval(); head.eval(); vl = []
        with torch.no_grad():
            for cid in val_c:
                d = _load(cid)
                if d is None or d.num_nodes == 0: continue
                emb = gnn(d.x, d.edge_index); vl.append(crit(head(emb).squeeze(-1), d.y).item())
        vl = float(np.mean(vl)) if vl else float("inf")
        print(f"[train_gnn] epoch {epoch+1}/{epochs} val_loss={vl:.4f}", flush=True)
        if vl < best_val - 1e-4:
            best_val = vl; best_g = {k: v.clone() for k, v in gnn.state_dict().items()}
            best_h = {k: v.clone() for k, v in head.state_dict().items()}; stall = 0
        else:
            stall += 1
            if stall >= patience: print(f"[train_gnn] early stop epoch {epoch+1}", flush=True); break
    if best_g: gnn.load_state_dict(best_g)
    if best_h: head.load_state_dict(best_h)
    config = {"in_channels": NODE_FEATURE_DIM, "hidden_channels": hidden_channels,
               "out_channels": GNN_OUTPUT_DIM, "num_layers": num_layers, "dropout": dropout}
    output_dir.mkdir(parents=True, exist_ok=True)
    save_gnn(gnn, config, output_dir)
    import joblib
    joblib.dump({"head": head.state_dict()}, output_dir / "gnn_head.joblib")
    metrics = {"best_val_loss": best_val}
    (output_dir / "gnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train_gnn] saved to {output_dir}", flush=True)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--files-parquet", type=Path, required=True)
    p.add_argument("--graphs-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    a = p.parse_args()
    train_gnn(a.files_parquet, a.graphs_dir, a.output_dir, epochs=a.epochs, patience=a.patience)

if __name__ == "__main__":
    main()
