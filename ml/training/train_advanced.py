"""Phase 7 ablation: metrics -> +CodeBERT -> +GRU history -> +GraphSAGE dependencies.

Every variant is trained and evaluated on exactly the same commits (those with
resolved source, a CodeBERT embedding, and a dependency graph) using the same
protocol as the deployed baseline: per-project temporal 60/20/20 split, Platt
calibration on validation probabilities, and a recall-aware F2 threshold chosen
on validation.  Model selection never looks at the test split.
"""

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from ml.features.schema import FEATURE_NAMES
from ml.training.advanced_model import FusionConfig, build_fusion_model, require_torch
from ml.training.train import (
    _metrics,
    _pipeline,
    select_operating_threshold,
    temporal_project_split,
)
from ml.training.train_codebert import (
    _calibrated_probability,
    _platt_calibrator,
    load_experiment_frame,
)

ADVANCED_MODEL_VERSION = "apachejit-fusion-ablation-v1"
NUMERIC_METRICS = ("precision", "recall", "f1", "f2", "roc_auc", "pr_auc")
MLP_VARIANTS: dict[str, dict[str, bool]] = {
    "mlp_metrics": {"use_codebert": False, "use_temporal": False, "use_graph": False},
    "mlp_metrics_codebert": {"use_codebert": True, "use_temporal": False, "use_graph": False},
    "mlp_metrics_codebert_gru": {"use_codebert": True, "use_temporal": True, "use_graph": False},
    "mlp_metrics_codebert_gru_gnn": {"use_codebert": True, "use_temporal": True, "use_graph": True},
}


@dataclass
class AblationInputs:
    frame: pd.DataFrame
    metrics: np.ndarray
    codebert: np.ndarray
    history: np.ndarray
    history_mask: np.ndarray
    graph_nodes: np.ndarray
    graph_edges: np.ndarray
    graph_mask: np.ndarray


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ablation_inputs(
    data_path: Path, embeddings_path: Path, histories_path: Path, graphs_path: Path
) -> AblationInputs:
    """Inner-join every representation on commit_id so all variants share rows."""
    frame, _ = load_experiment_frame(data_path, embeddings_path)
    with np.load(histories_path, allow_pickle=False) as bundle:
        history_index = {commit: row for row, commit in enumerate(_decode(bundle["commit_id"]))}
        history = np.asarray(bundle["history"], dtype=np.float32)
        history_mask = np.asarray(bundle["history_mask"], dtype=bool)
    with np.load(graphs_path, allow_pickle=False) as bundle:
        graph_index = {commit: row for row, commit in enumerate(_decode(bundle["commit_id"]))}
        graph_nodes = np.asarray(bundle["node_features"], dtype=np.float32)
        graph_edges = np.asarray(bundle["adjacency"], dtype=np.float32)
        graph_mask = np.asarray(bundle["node_mask"], dtype=bool)
    commits = frame["commit_id"].astype(str)
    frame = frame[commits.isin(history_index) & commits.isin(graph_index)].reset_index(drop=True)
    if frame.empty or frame["buggy"].nunique() < 2:
        raise ValueError("Ablation needs buggy and clean commits present in every input archive")
    embedding_columns = [column for column in frame.columns if column.startswith("emb_")]
    history_rows = [history_index[commit] for commit in frame["commit_id"].astype(str)]
    graph_rows = [graph_index[commit] for commit in frame["commit_id"].astype(str)]
    return AblationInputs(
        frame=frame.drop(columns=embedding_columns),
        metrics=frame[FEATURE_NAMES].to_numpy(dtype=np.float32),
        codebert=frame[embedding_columns].to_numpy(dtype=np.float32),
        history=history[history_rows],
        history_mask=history_mask[history_rows],
        graph_nodes=graph_nodes[graph_rows],
        graph_edges=graph_edges[graph_rows],
        graph_mask=graph_mask[graph_rows],
    )


class MetricScaler:
    """log1p + train-median imputation + train standardization for count-like metrics."""

    def fit(self, values: np.ndarray) -> "MetricScaler":
        logged = np.log1p(np.clip(values, 0, None))
        self.median = np.nanmedian(logged, axis=0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        filled = np.where(np.isnan(logged), self.median, logged)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale = np.where(self.scale < 1e-6, 1.0, self.scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        logged = np.log1p(np.clip(values, 0, None))
        filled = np.where(np.isnan(logged), self.median, logged)
        return ((filled - self.mean) / self.scale).astype(np.float32)


def _standardize_nodes(nodes: np.ndarray, mask: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    observed = nodes[train_rows][mask[train_rows]]
    mean = observed.mean(axis=0) if len(observed) else np.zeros(nodes.shape[-1])
    scale = observed.std(axis=0) if len(observed) else np.ones(nodes.shape[-1])
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (((nodes - mean) / scale) * mask[..., None]).astype(np.float32)


def _resolve_device(device: str) -> str:
    torch, _ = require_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _tensors(arrays: dict[str, np.ndarray], rows: np.ndarray, device: str) -> list:
    torch, _ = require_torch()
    order = ("metrics", "codebert", "history", "history_mask", "graph_nodes", "graph_edges")
    tensors = [torch.from_numpy(np.ascontiguousarray(arrays[key][rows])).to(device) for key in order]
    tensors.append(torch.from_numpy(np.ascontiguousarray(arrays["graph_mask"][rows])).to(device))
    return tensors


def _predict(model, tensors: list, batch_size: int = 1024) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            batch = [tensor[start : start + batch_size] for tensor in tensors]
            outputs.append(model.predict_probability(*batch).float().cpu().numpy())
    return np.concatenate(outputs)


def _train_fusion(
    config: FusionConfig,
    train_tensors: list,
    y_train: np.ndarray,
    validation_tensors: list,
    y_validation: np.ndarray,
    epochs: int,
    patience: int,
    seed: int,
    device: str,
    batch_size: int = 256,
):
    torch, nn = require_torch()
    torch.manual_seed(seed)
    model = build_fusion_model(config).to(device)
    positives = max(int(y_train.sum()), 1)
    pos_weight = torch.tensor((len(y_train) - positives) / positives, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    labels = torch.from_numpy(y_train.astype(np.float32)).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict | None = None
    best_score = -1.0
    stall = 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(labels), generator=generator).to(device)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(*[tensor[batch] for tensor in train_tensors]), labels[batch])
            loss.backward()
            optimizer.step()
        score = float(average_precision_score(y_validation, _predict(model, validation_tensors)))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_state is None:
        raise ValueError("Fusion model produced no improving epoch")
    model.load_state_dict(best_state)
    return model, best_state, best_score


def _evaluate(
    y_validation: np.ndarray, validation_probability: np.ndarray, y_test: np.ndarray, test_probability: np.ndarray
) -> tuple[dict, object, float]:
    calibrator = _platt_calibrator(pd.Series(y_validation), validation_probability)
    threshold = select_operating_threshold(
        pd.Series(y_validation), _calibrated_probability(calibrator, validation_probability)
    )
    metrics = _metrics(
        pd.Series(y_test), _calibrated_probability(calibrator, test_probability), threshold
    )
    return metrics, calibrator, threshold


def _summarize(runs: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for name in NUMERIC_METRICS:
        values = np.asarray([run["test"][name] for run in runs], dtype=float)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std())}
    validation = np.asarray([run["validation_pr_auc"] for run in runs], dtype=float)
    summary["validation_pr_auc"] = {"mean": float(validation.mean()), "std": float(validation.std())}
    return summary


def _log_mlflow(result: dict, output: Path) -> None:
    try:
        import mlflow
    except ImportError as error:
        print(f"[train_advanced] MLflow tracking skipped: {error}", flush=True)
        return
    with mlflow.start_run(run_name=ADVANCED_MODEL_VERSION):
        mlflow.log_params(
            {
                "split": result["split"],
                "seeds": ",".join(str(seed) for seed in result["config"]["seeds"]),
                "hidden_dim": result["config"]["hidden_dim"],
                "epochs": result["config"]["epochs"],
                "commits": result["subset"]["commits"],
                "data_sha256": result["dataset"]["data_sha256"],
                "embeddings_sha256": result["dataset"]["embeddings_sha256"],
            }
        )
        for variant, block in result["variants"].items():
            with mlflow.start_run(run_name=variant, nested=True):
                mlflow.log_params({"variant": variant, **block.get("inputs", {})})
                mlflow.log_metrics(
                    {f"{name}_mean": stats["mean"] for name, stats in block["summary"].items()}
                )
                mlflow.log_metrics(
                    {f"{name}_std": stats["std"] for name, stats in block["summary"].items()}
                )
        mlflow.log_artifact(str(output / "ablation.json"))


def train_advanced_experiment(
    data_path: Path,
    embeddings_path: Path,
    histories_path: Path,
    graphs_path: Path,
    output: Path,
    hidden_dim: int = 256,
    temporal_dim: int = 32,
    graph_dim: int = 32,
    epochs: int = 40,
    patience: int = 6,
    seeds: tuple[int, ...] = (42, 43, 44),
    device: str = "auto",
) -> dict:
    device = _resolve_device(device)
    inputs = load_ablation_inputs(data_path, embeddings_path, histories_path, graphs_path)
    frame = inputs.frame
    train_frame, validation_frame, test_frame = temporal_project_split(frame)
    if any(part["buggy"].nunique() < 2 for part in (train_frame, validation_frame, test_frame)):
        raise ValueError("Each temporal split must contain buggy and clean samples")
    position = {commit: index for index, commit in enumerate(frame["commit_id"].astype(str))}
    rows = {
        name: np.asarray([position[commit] for commit in part["commit_id"].astype(str)])
        for name, part in (("train", train_frame), ("validation", validation_frame), ("test", test_frame))
    }
    labels = {name: frame["buggy"].to_numpy()[index].astype(int) for name, index in rows.items()}

    scaler = MetricScaler().fit(inputs.metrics[rows["train"]])
    history = scaler.transform(inputs.history.reshape(-1, len(FEATURE_NAMES))).reshape(inputs.history.shape)
    arrays = {
        "metrics": scaler.transform(inputs.metrics),
        "codebert": inputs.codebert,
        "history": history * inputs.history_mask[..., None],
        "history_mask": inputs.history_mask,
        "graph_nodes": _standardize_nodes(inputs.graph_nodes, inputs.graph_mask, rows["train"]),
        "graph_edges": inputs.graph_edges,
        "graph_mask": inputs.graph_mask,
    }
    split_tensors = {name: _tensors(arrays, index, device) for name, index in rows.items()}
    base_config = FusionConfig(
        metric_dim=len(FEATURE_NAMES),
        codebert_dim=inputs.codebert.shape[1],
        node_dim=inputs.graph_nodes.shape[-1],
        temporal_dim=temporal_dim,
        graph_dim=graph_dim,
        hidden_dim=hidden_dim,
    )

    variants: dict[str, dict] = {}
    xgboost_runs = []
    for seed in seeds:
        baseline = _pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
        )
        baseline.fit(inputs.metrics[rows["train"]], labels["train"])
        validation_probability = baseline.predict_proba(inputs.metrics[rows["validation"]])[:, 1]
        test_probability = baseline.predict_proba(inputs.metrics[rows["test"]])[:, 1]
        metrics, _, _ = _evaluate(
            labels["validation"], validation_probability, labels["test"], test_probability
        )
        xgboost_runs.append(
            {
                "seed": seed,
                "validation_pr_auc": float(
                    average_precision_score(labels["validation"], validation_probability)
                ),
                "test": metrics,
            }
        )
    variants["xgboost_metrics"] = {
        "inputs": {"metrics": True, "codebert": False, "gru": False, "gnn": False},
        "runs": xgboost_runs,
        "summary": _summarize(xgboost_runs),
    }

    best: dict | None = None
    for variant, switches in MLP_VARIANTS.items():
        config = replace(base_config, **switches)
        runs = []
        for seed in seeds:
            model, state, validation_pr_auc = _train_fusion(
                config,
                split_tensors["train"],
                labels["train"],
                split_tensors["validation"],
                labels["validation"],
                epochs=epochs,
                patience=patience,
                seed=seed,
                device=device,
            )
            validation_probability = _predict(model, split_tensors["validation"])
            test_probability = _predict(model, split_tensors["test"])
            metrics, calibrator, threshold = _evaluate(
                labels["validation"], validation_probability, labels["test"], test_probability
            )
            runs.append({"seed": seed, "validation_pr_auc": validation_pr_auc, "test": metrics})
            if best is None or validation_pr_auc > best["validation_pr_auc"]:
                best = {
                    "variant": variant,
                    "seed": seed,
                    "validation_pr_auc": validation_pr_auc,
                    "config": config.to_dict(),
                    "torch_state": state,
                    "calibrator": calibrator,
                    "decision_threshold": threshold,
                }
        variants[variant] = {
            "inputs": {
                "metrics": True,
                "codebert": switches["use_codebert"],
                "gru": switches["use_temporal"],
                "gnn": switches["use_graph"],
            },
            "runs": runs,
            "summary": _summarize(runs),
        }

    ladder = ["mlp_metrics", *[name for name in MLP_VARIANTS if name != "mlp_metrics"]]
    deltas: dict[str, dict[str, float]] = {}
    for previous, current in itertools.pairwise(ladder):
        deltas[f"{current}_vs_{previous}"] = {
            name: round(
                variants[current]["summary"][name]["mean"] - variants[previous]["summary"][name]["mean"], 6
            )
            for name in NUMERIC_METRICS
        }
    for variant in MLP_VARIANTS:
        deltas[f"{variant}_vs_xgboost_metrics"] = {
            name: round(
                variants[variant]["summary"][name]["mean"]
                - variants["xgboost_metrics"]["summary"][name]["mean"],
                6,
            )
            for name in NUMERIC_METRICS
        }

    assert best is not None
    result = {
        "model_version": ADVANCED_MODEL_VERSION,
        "split": "60/20/20 temporal per project (same protocol as the metrics baseline)",
        "selection": "best MLP variant and seed by validation PR-AUC; test split never used",
        "config": {
            "hidden_dim": hidden_dim,
            "temporal_dim": temporal_dim,
            "graph_dim": graph_dim,
            "history_window": int(inputs.history.shape[1]),
            "max_graph_nodes": int(inputs.graph_nodes.shape[1]),
            "epochs": epochs,
            "patience": patience,
            "seeds": list(seeds),
            "device": device,
        },
        "dataset": {
            "data_sha256": _file_hash(data_path),
            "embeddings_sha256": _file_hash(embeddings_path),
        },
        "subset": {
            "projects": sorted(frame["project"].unique().tolist()),
            "commits": len(frame),
            "commits_per_project": {
                str(project): int(count) for project, count in frame["project"].value_counts().items()
            },
            "split_sizes": {name: len(index) for name, index in rows.items()},
            "buggy_rate": float(frame["buggy"].mean()),
            "commits_with_history": int(inputs.history_mask.any(axis=1).sum()),
            "graphs_with_edges": int((inputs.graph_edges.sum(axis=(1, 2)) > 0).sum()),
        },
        "variants": variants,
        "deltas": deltas,
        "best": {key: best[key] for key in ("variant", "seed", "validation_pr_auc")},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model_version": ADVANCED_MODEL_VERSION,
            "feature_names": FEATURE_NAMES,
            "metric_scaler": scaler,
            **best,
        },
        output / "advanced_model.joblib",
    )
    _log_mlflow(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Processed ApacheJIT CSV")
    parser.add_argument("--embeddings", type=Path, required=True, help="embeddings.npz")
    parser.add_argument("--inputs", type=Path, required=True, help="advanced.prepare output folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = train_advanced_experiment(
        args.data,
        args.embeddings,
        args.inputs / "temporal_histories.npz",
        args.inputs / "dependency_graphs.npz",
        args.output,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        patience=args.patience,
        seeds=tuple(int(seed) for seed in args.seeds.split(",") if seed.strip()),
        device=args.device,
    )
    print(
        json.dumps(
            {name: block["summary"] for name, block in result["variants"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
