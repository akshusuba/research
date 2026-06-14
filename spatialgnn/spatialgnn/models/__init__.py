"""Model zoo: the spatial GNN and the non-graph baselines it must beat."""

from .gnn import GNNNodeClassifier
from .mlp import MLPNodeClassifier

__all__ = ["GNNNodeClassifier", "MLPNodeClassifier"]
