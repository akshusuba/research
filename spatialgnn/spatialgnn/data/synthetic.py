"""Controlled spatial benchmark where the domain label is neighborhood-defined.

Each synthetic tissue ("slice") is a lattice of cells partitioned into
contiguous spatial domains (Voronoi regions around random centroids). Two
design choices make this a clean test of *where a GNN earns its keep*:

1. **Shared domain prototypes.** Domain ``d`` has the same latent expression
   prototype ``mu_d`` in every slice, so the (domain -> expression) mapping is
   learnable and transfers across slices -- a feature model is *not* doomed.

2. **Low per-cell signal, high spatial smoothness.** Each cell's expression is
   only weakly aligned to its prototype (``feature_snr``) and is further
   corrupted by dropout, so a single cell's features are an unreliable readout
   of its domain. But neighbouring cells share the domain, so *aggregating the
   neighbourhood* denoises the signal. That is exactly what a spatial GNN does
   and what an MLP/XGBoost (which see one cell at a time) cannot.

Crucially, coordinates are **not** given as features -- spatial information
enters only through the graph. Combined with cross-slice evaluation (domain
layouts differ per slice), this means absolute position cannot be memorized;
only neighbourhood-relative reasoning generalizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from ..config import SyntheticConfig


@dataclass
class SpatialDataset:
    data: Data                    # x, edge_index, y, pos, slice_id
    n_domains: int

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes


def _domain_prototypes(n_domains, n_genes, seed=12345):
    """Prototypes are shared across all slices (consistent domain semantics)."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 1.0, size=(n_domains, n_genes)).astype(np.float32)
    return mu


def _knn_edges(pos, k):
    n = pos.shape[0]
    k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(pos)
    _, idx = nn.kneighbors(pos)
    src = np.repeat(np.arange(n), k - 1)
    dst = idx[:, 1:].reshape(-1)
    # symmetrize
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    return np.stack([s, d])


def _make_slice(cfg: SyntheticConfig, mu, slice_seed):
    rng = np.random.default_rng(slice_seed)
    g = cfg.grid_size
    xs, ys = np.meshgrid(np.arange(g), np.arange(g))
    pos = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1).astype(np.float32)
    pos += rng.normal(0.0, cfg.jitter, size=pos.shape).astype(np.float32)
    n = pos.shape[0]

    # Contiguous domains via nearest random centroid (Voronoi).
    centroids = rng.uniform(0, g, size=(cfg.n_domains, 2)).astype(np.float32)
    d2 = ((pos[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    labels = d2.argmin(axis=1).astype(np.int64)

    # Expression: weak alignment to prototype + noise, then dropout.
    signal = mu[labels]                                   # (n, n_genes)
    noise = rng.normal(0.0, 1.0, size=signal.shape).astype(np.float32)
    expr = cfg.feature_snr * signal + (1.0 - cfg.feature_snr) * noise
    mask = rng.random(expr.shape) < cfg.dropout_rate
    expr[mask] = 0.0

    edges = _knn_edges(pos, cfg.k_neighbors)
    return pos, expr, labels, edges, n


def generate_synthetic_spatial(cfg: SyntheticConfig) -> SpatialDataset:
    """Generate ``cfg.n_slices`` tissues as one disjoint-union PyG graph."""
    mu = _domain_prototypes(cfg.n_domains, cfg.n_genes)
    all_pos, all_x, all_y, all_slice = [], [], [], []
    edge_src, edge_dst = [], []
    offset = 0
    for s in range(cfg.n_slices):
        pos, expr, labels, edges, n = _make_slice(cfg, mu, cfg.seed * 1000 + s)
        all_pos.append(pos)
        all_x.append(expr)
        all_y.append(labels)
        all_slice.append(np.full(n, s, dtype=np.int64))
        edge_src.append(edges[0] + offset)
        edge_dst.append(edges[1] + offset)
        offset += n

    x = torch.from_numpy(np.concatenate(all_x, axis=0))
    y = torch.from_numpy(np.concatenate(all_y, axis=0))
    pos = torch.from_numpy(np.concatenate(all_pos, axis=0))
    slice_id = torch.from_numpy(np.concatenate(all_slice, axis=0))
    edge_index = torch.from_numpy(
        np.stack([np.concatenate(edge_src), np.concatenate(edge_dst)])).long()

    data = Data(x=x, edge_index=edge_index, y=y, pos=pos, num_nodes=x.size(0))
    data.slice_id = slice_id
    return SpatialDataset(data=data, n_domains=cfg.n_domains)


def summarize(ds: SpatialDataset) -> dict:
    d = ds.data
    return {
        "num_cells": int(d.num_nodes),
        "num_slices": int(d.slice_id.max().item() + 1),
        "num_edges": int(d.edge_index.size(1) // 2),
        "n_domains": int(ds.n_domains),
        "n_genes": int(d.x.size(1)),
        "avg_degree": round(float(d.edge_index.size(1) / max(d.num_nodes, 1)), 2),
    }
