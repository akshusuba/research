"""Central configuration: paths, seeds, and default hyper-parameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

SEEDS: List[int] = [0, 1, 2, 42, 123]


@dataclass
class SyntheticConfig:
    """Controlled spatial benchmark where the domain label is neighborhood-defined.

    A tissue is tiled into contiguous spatial ``domains``. Each cell sits at a
    2D location and belongs to the domain of its region. The catch: a cell's
    *expression features* only weakly indicate its domain (low ``feature_snr``)
    and are corrupted by dropout, while the domain is spatially smooth. So the
    reliable way to recover a cell's domain is to look at its spatial neighbors
    -- exactly the signal a spatial GNN aggregates and an MLP cannot see.

    Multiple independent ``n_slices`` tissues are generated so we can evaluate
    cross-slice generalization (absolute coordinates do not transfer).
    """

    n_slices: int = 6
    grid_size: int = 28           # cells arranged on a grid_size x grid_size lattice
    n_domains: int = 5
    n_genes: int = 50             # expression feature dimension
    feature_snr: float = 0.35     # how much a cell's own expression reveals its domain
    dropout_rate: float = 0.4     # fraction of expression entries zeroed (technical dropout)
    jitter: float = 0.3           # positional jitter (cells are not on a perfect grid)
    k_neighbors: int = 6          # spatial graph connectivity
    seed: int = 0


@dataclass
class ModelConfig:
    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    encoder: str = "sage"         # "sage" | "gat" | "gcn" | "mlp"


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 5e-3
    weight_decay: float = 5e-4
    patience: int = 40
    device: str = "cpu"
    verbose: bool = False


@dataclass
class SplitConfig:
    # Cross-slice: hold out whole tissues for val/test (absolute position can't
    # transfer, so only neighborhood-relative reasoning generalizes).
    mode: str = "cross_slice"     # "cross_slice" | "within_slice"
    val_slices: int = 1
    test_slices: int = 2
    seed: int = 0


@dataclass
class ExperimentConfig:
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def to_dict(self) -> dict:
        return asdict(self)
