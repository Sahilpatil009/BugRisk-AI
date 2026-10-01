"""Fusion model inference service for the BugRisk AI worker."""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Any
import numpy as np
from .risk import FEATURE_NAMES

logger = logging.getLogger(__name__)

CODEBERT_DIM = 768
GRU_DIM = 128
GNN_DIM = 64
METRICS_DIM = 11
FUSION_INPUT_DIM = CODEBERT_DIM + GRU_DIM + GNN_DIM + METRICS_DIM  # 971


def _row_key(commit_id: str, file_path: str) -> str:
    return hashlib.sha256(f"{commit_id}\x00{file_path}".encode()).hexdigest()[:32]


class FusionModelService:
    """Loads the fusion bundle and runs per-file inference."""

    def __init__(self, bundle_path: Path) -> None:
        self.bundle_path = bundle_path
        self.bundle: dict[str, Any] | None = None
        self._mlp: Any = None
        self._calibrator: Any = None
        self._torch: Any = None
        if not bundle_path.exists():
            return
        try:
            import joblib, torch
            self._torch = torch
            self.bundle = joblib.load(bundle_path)
            if self.bundle.get("model_type") != "fusion-mlp-v1":
                logger.warning("fusion bundle model_type is not fusion-mlp-v1; skipping")
                self.bundle = None
                return
            from ml.models.fusion_mlp import build_fusion_mlp
            config = self.bundle.get("fusion_config", {})
            self._mlp = build_fusion_mlp(
                input_dim=config.get("input_dim", FUSION_INPUT_DIM),
                dropout1=config.get("dropout1", 0.3),
                dropout2=config.get("dropout2", 0.2),
            )
            mlp_weights_bytes = self.bundle.get("mlp_weights")
            if mlp_weights_bytes:
                import io
                self._mlp.load_state_dict(torch.load(io.BytesIO(mlp_weights_bytes), map_location="cpu"))
            self._mlp.eval()
            self._calibrator = self.bundle.get("calibrator")
        except Exception:
            logger.exception("Failed to load fusion model bundle")
            self.bundle = None

    @property
    def available(self) -> bool:
        return self.bundle is not None and self._mlp is not None

    def predict_file(self, source: str, history_sequence: np.ndarray,
                      graph_embedding: np.ndarray, metrics: dict[str, float]) -> float:
        """Return calibrated bug probability for a single file."""
        if not self.available:
            raise RuntimeError("Fusion model is not loaded")
        torch = self._torch
        codebert_vec = np.zeros(CODEBERT_DIM, dtype=np.float32)
        try:
            from ml.embeddings.embed import CodeBERTEncoder
            codebert_vec = CodeBERTEncoder().encode([source])[0]
        except Exception:
            pass
        gru_vec = np.zeros(GRU_DIM, dtype=np.float32)
        try:
            gru_weights_bytes = self.bundle.get("gru_weights") if self.bundle else None
            if gru_weights_bytes:
                import io
                from ml.models.gru_history import build_gru, HISTORY_FEATURES
                cfg = self.bundle.get("gru_config", {}) if self.bundle else {}
                gru = build_gru(input_size=cfg.get("input_size", len(HISTORY_FEATURES)),
                                hidden_size=cfg.get("hidden_size", GRU_DIM),
                                num_layers=cfg.get("num_layers", 2),
                                dropout=cfg.get("dropout", 0.2))
                gru.load_state_dict(torch.load(io.BytesIO(gru_weights_bytes), map_location="cpu"))
                gru.eval()
                with torch.no_grad():
                    _, hid = gru(torch.tensor(history_sequence[np.newaxis], dtype=torch.float32))
                gru_vec = hid[-1][0].numpy()
        except Exception:
            pass
        gnn_vec = (graph_embedding[:GNN_DIM] if len(graph_embedding) >= GNN_DIM
                   else np.pad(graph_embedding, (0, GNN_DIM - len(graph_embedding))))
        metric_vec = np.array([float(metrics.get(n, 0.0)) for n in FEATURE_NAMES[:METRICS_DIM]], dtype=np.float32)
        x = torch.tensor(np.concatenate([codebert_vec, gru_vec, gnn_vec, metric_vec])[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            logit = float(self._mlp(x).squeeze().item())
        if self._calibrator is not None:
            prob = float(self._calibrator.predict_proba([[logit]])[0, 1])
        else:
            prob = float(1.0 / (1.0 + np.exp(-logit)))
        return round(max(0.0, min(1.0, prob)), 4)

    @property
    def model_version(self) -> str:
        return str(self.bundle.get("model_version", "fusion-mlp-v1")) if self.bundle else "fusion-mlp-v1"