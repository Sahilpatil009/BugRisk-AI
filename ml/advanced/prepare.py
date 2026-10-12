"""Build GRU histories and per-commit dependency graphs for the research model.

Every history row contains only metrics of commits with a strictly earlier
timestamp in the same project.  Dependency graphs are constructed solely from
the files changed by that commit, so neither representation reads future
labels or source.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features.schema import FEATURE_NAMES

IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+(\.*[\w.]*)\s+import|import\s+(?:static\s+)?([\w.]+)(?:\.?\{([^}]*)\})?)",
    re.MULTILINE,
)
BRANCH_PATTERN = re.compile(r"\b(?:if|for|while|case|catch|except)\b|&&|\|\|")
DECLARATION_PATTERN = re.compile(r"\b(?:def|class|function|void|public|private)\b")
SOURCE_ROOTS = ("src/main/java/", "src/test/java/", "src/main/scala/", "src/test/scala/", "src/")
NODE_FEATURE_DIM = 5


def build_temporal_histories(
    frame: pd.DataFrame, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return left-padded prior-metric sequences, masks, commit ids, and projects."""
    if window < 1:
        raise ValueError("window must be at least one")
    ordered = frame.sort_values(["project", "event_time", "commit_id"]).reset_index(drop=True)
    histories = np.zeros((len(ordered), window, len(FEATURE_NAMES)), dtype=np.float32)
    masks = np.zeros((len(ordered), window), dtype=bool)
    offset = 0
    for _, project in ordered.groupby("project", sort=False):
        values = project[FEATURE_NAMES].to_numpy(dtype=np.float32)
        times = project["event_time"].to_numpy()
        # Commits sharing a timestamp are not ordered, so none may see the others.
        first_at_time = np.searchsorted(times, times, side="left")
        for index, end in enumerate(first_at_time):
            history = values[max(0, end - window) : end]
            if len(history):
                histories[offset + index, -len(history) :] = history
                masks[offset + index, -len(history) :] = True
        offset += len(project)
    return (
        histories,
        masks,
        ordered["commit_id"].astype(str).to_numpy(dtype="U64"),
        ordered["project"].astype(str).to_numpy(dtype="U96"),
    )


def _node_features(content: str) -> np.ndarray:
    """Small static source representation for a changed-file graph node."""
    lines = content.count("\n") + 1
    imports = len(IMPORT_PATTERN.findall(content))
    branches = len(BRANCH_PATTERN.findall(content))
    declarations = len(DECLARATION_PATTERN.findall(content))
    return np.log1p(
        np.asarray([lines, len(content), imports, branches, declarations], dtype=np.float32)
    )


def _module_name(path: str) -> str:
    clean = path.replace("\\", "/")
    for root in SOURCE_ROOTS:
        position = clean.find(root)
        if position != -1:
            clean = clean[position + len(root) :]
            break
    return clean.rsplit(".", 1)[0].replace("/", ".").lower()


def _module_keys(path: str) -> set[str]:
    """Dotted suffixes of a file's module name, e.g. org.a.B -> {org.a.b, a.b, b}."""
    parts = _module_name(path).split(".")
    return {".".join(parts[index:]) for index in range(len(parts))}


def _imported_names(content: str) -> list[str]:
    names: list[str] = []
    for python_from, imported, members in IMPORT_PATTERN.findall(content):
        if python_from:
            names.append(python_from.lstrip("."))
        elif members:
            base = imported.rstrip(".")
            names.extend(
                f"{base}.{member.split('=>')[0].strip()}" for member in members.split(",")
            )
        else:
            names.append(imported.rstrip(".*").rstrip("."))
    return [name.lower() for name in names if name]


def dependency_adjacency(files: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Infer directed import edges among the source files changed in one commit."""
    ordered = files.sort_values("file_path").reset_index(drop=True)
    paths = ordered["file_path"].astype(str).tolist()
    features = np.vstack([_node_features(str(content)) for content in ordered["content"]])
    adjacency = np.zeros((len(paths), len(paths)), dtype=np.float32)
    module_index: dict[str, int] = {}
    for index, path in enumerate(paths):
        for key in _module_keys(path):
            module_index.setdefault(key, index)
    for source_index, content in enumerate(ordered["content"].astype(str)):
        for imported in _imported_names(content):
            # Static imports name a member; walk up until a changed module matches.
            parts = imported.split(".")
            for length in range(len(parts), 0, -1):
                target = module_index.get(".".join(parts[:length]))
                if target is not None:
                    if target != source_index:
                        adjacency[source_index, target] = 1.0
                    break
                if length <= 2:
                    break
    return features, adjacency, paths


def build_graph_archive(sources: pd.DataFrame, output: Path, max_nodes: int = 8) -> dict:
    """Persist padded changed-file graphs aligned by commit id for Torch training."""
    required = {"project", "commit_id", "file_path", "content"}
    missing = required - set(sources.columns)
    if missing:
        raise ValueError(f"Source data missing columns: {sorted(missing)}")
    records: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for (project, commit_id), files in sources.groupby(["project", "commit_id"], sort=True):
        features, adjacency, _ = dependency_adjacency(files)
        count = min(len(features), max_nodes)
        if count:
            records.append(
                (str(project), str(commit_id), features[:count], adjacency[:count, :count])
            )
    if not records:
        raise ValueError("No dependency graphs could be created from the source data")
    nodes = np.zeros((len(records), max_nodes, NODE_FEATURE_DIM), dtype=np.float32)
    adjacency = np.zeros((len(records), max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros((len(records), max_nodes), dtype=bool)
    for index, (_, _, features, edges) in enumerate(records):
        count = len(features)
        nodes[index, :count] = features
        adjacency[index, :count, :count] = edges
        mask[index, :count] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        commit_id=np.asarray([record[1] for record in records], dtype="U64"),
        project=np.asarray([record[0] for record in records], dtype="U96"),
        node_features=nodes,
        adjacency=adjacency,
        node_mask=mask,
    )
    return {
        "graphs": len(records),
        "graphs_with_edges": int((adjacency.sum(axis=(1, 2)) > 0).sum()),
        "max_nodes": max_nodes,
        "node_feature_dim": NODE_FEATURE_DIM,
        "projects": sorted({record[0] for record in records}),
    }


def prepare_advanced_inputs(
    data_path: Path, sources_path: Path, output: Path, window: int = 5, max_nodes: int = 8
) -> dict:
    sources = pd.read_parquet(sources_path)
    frame = pd.read_csv(data_path, parse_dates=["event_time"])
    frame = frame[frame["project"].isin(set(sources["project"]))]
    histories, masks, commit_ids, projects = build_temporal_histories(frame, window)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "temporal_histories.npz",
        commit_id=commit_ids,
        project=projects,
        history=histories,
        history_mask=masks,
    )
    graph_meta = build_graph_archive(sources, output / "dependency_graphs.npz", max_nodes)
    metadata = {
        "history_window": window,
        "history_feature_dim": len(FEATURE_NAMES),
        "histories": len(commit_ids),
        **graph_meta,
    }
    (output / "advanced_inputs.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=8)
    args = parser.parse_args()
    metadata = prepare_advanced_inputs(
        args.data, args.sources, args.output, window=args.window, max_nodes=args.max_nodes
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
