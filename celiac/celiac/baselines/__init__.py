"""
Baseline models for knowledge graph link prediction.

Includes:
- KGE models: TransE, DistMult, RotatE, ComplEx
- Non-graph: node2vec
- GNN baselines: R-GCN, CompGCN, HGT
"""

from .kge_models import (
    TransEModel,
    DistMultModel,
    RotatEModel,
    ComplExModel,
    KGELinkPredictor,
)
from .gnn_baselines import (
    RGCNModel,
    CompGCNModel,
    HGTModel,
)
from .node2vec_baseline import Node2VecModel
from .trainer import BaselineTrainer, BASELINE_REGISTRY

__all__ = [
    # KGE models
    'TransEModel',
    'DistMultModel',
    'RotatEModel',
    'ComplExModel',
    'KGELinkPredictor',
    # GNN baselines
    'RGCNModel',
    'CompGCNModel',
    'HGTModel',
    # Non-graph
    'Node2VecModel',
    # Trainer
    'BaselineTrainer',
    'BASELINE_REGISTRY',
]
