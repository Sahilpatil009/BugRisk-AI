"""Parse Python import relationships to build per-commit dependency graphs."""
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from ml.features.schema import FEATURE_NAMES

METRIC_COLS = FEATURE_NAMES


def _parse_imports(source: str) -> list[str]:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    return imports


def _build_commit_graph(commit_files: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_ids = commit_files["file_path"].tolist()
    node_to_idx = {path: idx for idx, path in enumerate(node_ids)}
    edges_src: list[int] = []
    edges_dst: list[int] = []
    for _, row in commit_files.iterrows():
        if not str(row.get("file_path", "")).endswith(".py"):
            continue
        for mod in _parse_imports(str(row.get("content", ""))):
            candidate = f"{mod}.py"
            if candidate in node_to_idx:
                src = node_to_idx[str(row["file_path"])]
                dst = node_to_idx[candidate]
                if src != dst:
                    edges_src.append(src); edges_dst.append(dst)
    edge_index = (np.array([edges_src, edges_dst], dtype=np.int64)
                  if edges_src else np.zeros((2, 0), dtype=np.int64))
    feat_cols = [c for c in METRIC_COLS if c in commit_files.columns]
    node_features = (commit_files[feat_cols].to_numpy(dtype=np.float32)
                     if feat_cols else np.zeros((len(node_ids), 1), dtype=np.float32))
    return np.array(node_ids, dtype="U"), edge_index, node_features


def build_dependency_graphs(files_parquet: Path, output_dir: Path) -> None:
    frame = pd.read_parquet(files_parquet)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit_ids = frame["commit_id"].unique()
    print(f"[dependency_graphs] building graphs for {len(commit_ids)} commits", flush=True)
    meta: list[dict] = []
    for cid in commit_ids:
        cf = frame[frame["commit_id"] == cid]
        node_ids, edge_index, node_features = _build_commit_graph(cf)
        safe = hashlib.sha256(str(cid).encode()).hexdigest()[:16]
        npz_path = output_dir / f"{safe}.npz"
        np.savez_compressed(npz_path, node_ids=node_ids, edge_index=edge_index, node_features=node_features)
        meta.append({"commit_id": str(cid), "file": npz_path.name,
                      "n_nodes": len(node_ids), "n_edges": int(edge_index.shape[1])})
    (output_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[dependency_graphs] done. {len(meta)} commit graphs saved.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--files-parquet", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    build_dependency_graphs(a.files_parquet, a.output_dir)

if __name__ == "__main__":
    main()
