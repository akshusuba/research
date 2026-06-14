"""Central configuration: paths, seeds, and default hyper-parameters.

Everything is plain dataclasses so configs are easy to serialize into the
results JSON for reproducibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# Multi-seed protocol (same rigor bar as the celiac project).
SEEDS: List[int] = [0, 1, 2, 42, 123]


@dataclass
class SyntheticConfig:
    """Controlled benchmark where SL is *defined* by graph topology.

    Genes are organized into `n_processes` essential biological processes.
    Each process is implemented by `modules_per_process` redundant pathway
    modules. Genes inside a module are densely interconnected (they cooperate);
    a gene pair is synthetic-lethal iff the two genes live in *different*
    redundant modules of the *same* process -- knocking out both removes all
    redundancy for that essential process. Same-module pairs are NOT lethal
    (still redundant), and cross-process pairs are NOT lethal (unrelated).

    Crucially, node features carry no SL signal (Gaussian noise + degree), so
    the only way to predict SL is to read the graph topology. This isolates
    the contribution of structure.
    """

    n_processes: int = 40
    modules_per_process: int = 3
    genes_per_module: int = 8
    intra_module_p: float = 0.6      # edge prob within a redundant module
    inter_module_p: float = 0.04     # edge prob between modules of same process
    cross_process_p: float = 0.002   # background edges across processes
    noise_features: int = 16         # dim of uninformative node features
    feature_signal: float = 0.0      # 0.0 => features are pure noise (no SL info)
    seed: int = 0


@dataclass
class ModelConfig:
    hidden_channels: int = 128
    out_channels: int = 64
    # 3 hops let a gene reach other redundant modules of the same process;
    # 4+ layers over-smooth and collapse (see scripts/diagnose.py).
    num_layers: int = 3
    dropout: float = 0.2
    encoder: str = "sage"            # "sage" | "gat" | "gcn"
    # SL is a "same-process, different-module" relationship -- a moderate
    # embedding-distance band, not raw similarity -- so an expressive MLP
    # decoder (shared with the node2vec baseline for fairness) is the default.
    decoder: str = "mlp"             # "mlp" | "bilinear" | "dot"


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 2e-3
    weight_decay: float = 5e-4
    patience: int = 40               # early-stopping patience (val AUPRC)
    neg_ratio: int = 1               # negatives per positive
    # Training a GNN from uninformative features must break symmetry to escape
    # the chance plateau; a few restarts (best val kept) removes init-luck.
    num_restarts: int = 3
    device: str = "cpu"
    verbose: bool = False


@dataclass
class SplitConfig:
    val_frac: float = 0.15
    test_frac: float = 0.15
    mode: str = "transductive"       # "transductive" | "inductive"
    seed: int = 0


@dataclass
class ExperimentConfig:
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def to_dict(self) -> dict:
        return asdict(self)
