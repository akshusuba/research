"""Leakage-safe spatial splits for node classification.

* ``cross_slice`` (default, decisive) -- whole tissues are assigned to
  train/val/test. Because synthetic slices (and real tissue sections) have
  *different* domain layouts, absolute coordinates do not transfer; a model
  must learn the neighbourhood-relative (domain) structure to generalize. In
  the disjoint-union graph, train and test slices are separate connected
  components, so message passing cannot leak labels across the split.

* ``within_slice`` -- spatially-blocked split *inside* each slice (contiguous
  blocks held out), a within-tissue generalization test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import SplitConfig


@dataclass
class SpatialSplit:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_classes: int

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)


def make_split(ds, cfg: SplitConfig) -> SpatialSplit:
    rng = np.random.default_rng(cfg.seed)
    data = ds.data
    if cfg.mode == "cross_slice":
        train_m, val_m, test_m = _cross_slice(data, cfg, rng)
    elif cfg.mode == "within_slice":
        train_m, val_m, test_m = _within_slice(data, cfg, rng)
    else:
        raise ValueError(cfg.mode)
    return SpatialSplit(
        x=data.x, edge_index=data.edge_index, y=data.y,
        train_mask=train_m, val_mask=val_m, test_mask=test_m,
        num_classes=int(data.y.max().item() + 1),
    )


def _mask(n, idx):
    m = torch.zeros(n, dtype=torch.bool)
    m[idx] = True
    return m


def _cross_slice(data, cfg, rng):
    slice_id = data.slice_id.numpy()
    slices = np.unique(slice_id)
    rng.shuffle(slices)
    n_test = cfg.test_slices
    n_val = cfg.val_slices
    test_s = set(slices[:n_test])
    val_s = set(slices[n_test:n_test + n_val])
    n = data.num_nodes
    test_idx = np.where(np.isin(slice_id, list(test_s)))[0]
    val_idx = np.where(np.isin(slice_id, list(val_s)))[0]
    train_idx = np.where(~np.isin(slice_id, list(test_s | val_s)))[0]
    return _mask(n, train_idx), _mask(n, val_idx), _mask(n, test_idx)


def _within_slice(data, cfg, rng):
    """Hold out contiguous spatial blocks within every slice."""
    pos = data.pos.numpy()
    slice_id = data.slice_id.numpy()
    n = data.num_nodes
    train_idx, val_idx, test_idx = [], [], []
    for s in np.unique(slice_id):
        idx = np.where(slice_id == s)[0]
        # block by a random axis threshold on x-coordinate (contiguous regions)
        coord = pos[idx, 0]
        order = np.argsort(coord)
        nse = len(idx)
        cut_test = int(0.7 * nse)
        cut_val = int(0.85 * nse)
        test_idx += list(idx[order[cut_val:]])
        val_idx += list(idx[order[cut_test:cut_val]])
        train_idx += list(idx[order[:cut_test]])
    return (_mask(n, np.array(train_idx)), _mask(n, np.array(val_idx)),
            _mask(n, np.array(test_idx)))
