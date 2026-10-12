"""Portable PyTorch GRU + GraphSAGE fusion encoder used by the offline research run."""

from dataclasses import asdict, dataclass


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Advanced research requires the 'codebert' extra: pip install -e 'backend[codebert]'"
        ) from error
    return torch, nn


@dataclass(frozen=True)
class FusionConfig:
    metric_dim: int
    codebert_dim: int = 768
    node_dim: int = 5
    temporal_dim: int = 32
    graph_dim: int = 32
    hidden_dim: int = 256
    use_codebert: bool = True
    use_temporal: bool = True
    use_graph: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fused_dim(self) -> int:
        return (
            self.metric_dim
            + (self.codebert_dim if self.use_codebert else 0)
            + (self.temporal_dim if self.use_temporal else 0)
            + (self.graph_dim if self.use_graph else 0)
        )


def build_fusion_model(config: FusionConfig):
    """Build a metrics [+ CodeBERT] [+ GRU history] [+ GraphSAGE] commit classifier.

    Disabled branches are not instantiated, so every ablation variant has only
    the parameters it actually uses.
    """
    torch, nn = require_torch()

    class GraphSAGE(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.self_layer = nn.Linear(config.node_dim, config.graph_dim)
            self.neighbor_layer = nn.Linear(config.node_dim, config.graph_dim, bias=False)

        def forward(self, node_features, adjacency, node_mask):
            mask = node_mask.unsqueeze(-1).to(node_features.dtype)
            # Import edges are directed; risk can flow both ways along a dependency.
            adjacency = torch.clamp(adjacency + adjacency.transpose(1, 2), max=1.0)
            adjacency = adjacency * mask * mask.transpose(1, 2)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
            neighbor_mean = adjacency @ node_features / degree
            encoded = torch.relu(
                self.self_layer(node_features) + self.neighbor_layer(neighbor_mean)
            )
            return (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    class FusionModel(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            # The extra input channel is the history mask, so padding is distinguishable
            # from a real prior commit whose standardized metrics happen to be zero.
            self.gru = (
                nn.GRU(config.metric_dim + 1, config.temporal_dim, batch_first=True)
                if config.use_temporal
                else None
            )
            self.graph = GraphSAGE() if config.use_graph else None
            self.classifier = nn.Sequential(
                nn.Linear(config.fused_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(
            self, metrics, codebert, history, history_mask, graph_nodes, graph_edges, graph_mask
        ):
            parts = [metrics]
            if config.use_codebert:
                parts.append(codebert)
            if self.gru is not None:
                mask = history_mask.unsqueeze(-1).to(history.dtype)
                sequence, _ = self.gru(torch.cat([history * mask, mask], dim=-1))
                # Histories are left-padded, so the final step is the most recent commit.
                temporal = sequence[:, -1]
                parts.append(temporal * history_mask.any(dim=1, keepdim=True).to(temporal.dtype))
            if self.graph is not None:
                parts.append(self.graph(graph_nodes, graph_edges, graph_mask))
            return self.classifier(torch.cat(parts, dim=1)).squeeze(1)

        def predict_probability(self, *inputs):
            """Convert the final MLP logit into the PRD's 0-1 risk probability."""
            return torch.sigmoid(self(*inputs))

    return FusionModel()
