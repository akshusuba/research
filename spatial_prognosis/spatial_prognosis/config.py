"""Central configuration: paths, seeds, and default hyper-parameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List

try:
    import torch
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEFAULT_DEVICE = "cpu"

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
    """Patient cohort where outcome is set by spatial ARRANGEMENT, not composition.

    Each patient is one tumor section represented as a cell graph. Two outcome
    classes are generated with *identical cell-type composition* (the same number
    of tumor and immune cells) but opposite spatial organization:

    * ``infiltrated`` (favorable): immune cells are dispersed among tumor cells
      -> many tumor-immune spatial contacts.
    * ``excluded`` (unfavorable): immune cells are segregated into their own
      region -> few tumor-immune contacts.

    Because composition is matched, a model that sees only cell-type proportions
    (XGBoost/MLP on aggregated features) cannot separate the classes; only a
    model that reads the spatial graph can. This mirrors the real, prognostically
    important "immune infiltration vs. exclusion" phenotype.
    """

    n_patients: int = 300
    cells_per_patient: int = 350
    immune_fraction: float = 0.35     # identical across both classes
    field_size: float = 30.0          # tissue extent (microns, arbitrary)
    k_neighbors: int = 6              # spatial graph connectivity
    marker_dim: int = 8               # noisy per-cell marker features
    marker_snr: float = 0.5           # how much markers reveal cell type
    exclusion_strength: float = 0.85  # how strongly immune cells segregate (excluded class)
    label_noise: float = 0.05         # fraction of flipped labels (realism)
    seed: int = 0


@dataclass
class ModelConfig:
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    encoder: str = "sage"             # "sage" | "gin" | "gcn"
    pool: str = "mean"                # graph readout: "mean" | "max"


@dataclass
class TrainConfig:
    epochs: int = 150
    lr: float = 5e-3
    weight_decay: float = 5e-4
    patience: int = 30
    batch_size: int = 32
    device: str = DEFAULT_DEVICE   # auto: "cuda" if a GPU is visible, else "cpu"
    verbose: bool = False


@dataclass
class SplitConfig:
    val_frac: float = 0.15
    test_frac: float = 0.20
    seed: int = 0


@dataclass
class ExperimentConfig:
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def to_dict(self) -> dict:
        return asdict(self)
