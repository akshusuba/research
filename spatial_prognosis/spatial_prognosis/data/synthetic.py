"""Synthetic patient cohort: outcome determined by spatial ARRANGEMENT.

Each patient is one tumor section, represented as a graph of cells (nodes) with
2D positions and per-cell features, connected by a spatial kNN graph. Two
outcome classes are generated with *matched cell-type composition* but opposite
spatial organization of immune cells:

* class 1 -- ``infiltrated`` (favorable): immune cells dispersed among tumor
  -> many tumor-immune spatial contacts.
* class 0 -- ``excluded`` (unfavorable): immune cells segregated into a niche
  -> few tumor-immune contacts.

Both classes have the SAME immune fraction and the SAME per-cell feature
distribution, so a model that sees only cell-type proportions / aggregated
features (XGBoost, MLP) is at chance; only a model that reads the spatial graph
can recover the label. This is the controlled, biologically-motivated proof that
spatial arrangement carries signal beyond composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from ..config import SyntheticConfig

TUMOR, IMMUNE = 0, 1


@dataclass
class CohortDataset:
    graphs: List[Data]
    num_classes: int
    num_features: int

    def __len__(self):
        return len(self.graphs)


def _knn_edges(pos, k):
    n = pos.shape[0]
    k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(pos)
    _, idx = nn.kneighbors(pos)
    src = np.repeat(np.arange(n), k - 1)
    dst = idx[:, 1:].reshape(-1)
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    return torch.from_numpy(np.stack([s, d])).long()


def _make_patient(cfg: SyntheticConfig, label: int, mu, rng) -> Data:
    n = cfg.cells_per_patient
    n_immune = int(cfg.immune_fraction * n)
    n_tumor = n - n_immune
    f = cfg.field_size

    tumor_pos = rng.uniform(0, f, size=(n_tumor, 2))
    if label == 1:  # infiltrated: immune dispersed uniformly among tumor
        immune_pos = rng.uniform(0, f, size=(n_immune, 2))
    else:           # excluded: immune segregated into a corner niche
        in_niche = rng.random(n_immune) < cfg.exclusion_strength
        niche = rng.normal(loc=[0.15 * f, 0.15 * f], scale=0.10 * f,
                           size=(n_immune, 2))
        unif = rng.uniform(0, f, size=(n_immune, 2))
        immune_pos = np.where(in_niche[:, None], niche, unif)
        immune_pos = np.clip(immune_pos, 0, f)

    pos = np.vstack([tumor_pos, immune_pos]).astype(np.float32)
    types = np.array([TUMOR] * n_tumor + [IMMUNE] * n_immune, dtype=np.int64)

    # per-cell features: cell-type one-hot + noisy markers (markers depend ONLY
    # on cell type, so the feature distribution is identical across classes).
    onehot = np.zeros((n, 2), dtype=np.float32)
    onehot[np.arange(n), types] = 1.0
    noise = rng.normal(0, 1, size=(n, cfg.marker_dim)).astype(np.float32)
    markers = cfg.marker_snr * mu[types] + (1 - cfg.marker_snr) * noise
    x = np.concatenate([onehot, markers], axis=1).astype(np.float32)

    edge_index = _knn_edges(pos, cfg.k_neighbors)
    data = Data(x=torch.from_numpy(x), edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.long),
                pos=torch.from_numpy(pos))
    data.cell_type = torch.from_numpy(types)
    return data


def generate_cohort(cfg: SyntheticConfig) -> CohortDataset:
    rng = np.random.default_rng(cfg.seed)
    mu = np.random.default_rng(777).normal(0, 1, size=(2, cfg.marker_dim)).astype(np.float32)
    graphs: List[Data] = []
    for i in range(cfg.n_patients):
        label = i % 2  # balanced
        if rng.random() < cfg.label_noise:
            label = 1 - label
        graphs.append(_make_patient(cfg, label, mu, rng))
    return CohortDataset(graphs=graphs, num_classes=2,
                         num_features=graphs[0].x.size(1))


def composition_features(graphs: List[Data]) -> np.ndarray:
    """Per-patient 'bag-of-cells' features: what a composition-only model sees.

    Cell-type proportions + mean/std of marker features over the patient's
    cells. Deliberately contains NO spatial-arrangement information.
    """
    feats = []
    for g in graphs:
        x = g.x.numpy()
        onehot, markers = x[:, :2], x[:, 2:]
        prop = onehot.mean(axis=0)                 # cell-type proportions
        m_mean = markers.mean(axis=0)
        m_std = markers.std(axis=0)
        feats.append(np.concatenate([prop, m_mean, m_std]))
    return np.asarray(feats, dtype=np.float32)


def summarize(ds: CohortDataset) -> dict:
    ys = np.array([int(g.y) for g in ds.graphs])
    # average tumor-immune contact rate by class (diagnostic of the mechanism)
    def contact_rate(g):
        ei = g.edge_index.numpy(); t = g.cell_type.numpy()
        cross = (t[ei[0]] != t[ei[1]]).mean()
        return cross
    rates = np.array([contact_rate(g) for g in ds.graphs])
    return {
        "num_patients": len(ds.graphs),
        "cells_per_patient": int(ds.graphs[0].x.size(0)),
        "num_features": ds.num_features,
        "class_balance": [int((ys == 0).sum()), int((ys == 1).sum())],
        "tumor_immune_contact_excluded": round(float(rates[ys == 0].mean()), 3),
        "tumor_immune_contact_infiltrated": round(float(rates[ys == 1].mean()), 3),
    }
