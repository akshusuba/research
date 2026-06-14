"""Model zoo: the GNN (our model) and the non-graph baselines it must beat."""

from .gnn import GNNLinkPredictor
from .mlp import MLPLinkPredictor
from .node2vec import Node2VecLinkPredictor

__all__ = ["GNNLinkPredictor", "MLPLinkPredictor", "Node2VecLinkPredictor"]
