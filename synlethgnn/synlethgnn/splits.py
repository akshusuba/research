"""Train/val/test splitting for synthetic-lethality link prediction.

Two regimes are supported, and the difference between them is central to the
project's thesis:

* ``transductive``  -- positive/negative *pairs* are split randomly. Every gene
  is seen during training. node2vec and KGE baselines can do well here because
  they only need to memorize a good embedding per (seen) node.

* ``inductive`` (cold-gene) -- a fraction of *genes* is held out entirely.
  Test/val pairs involve at least one never-seen gene, and held-out genes are
  removed from the message-passing graph used during training. At evaluation
  the full graph is restored so held-out genes can attach to the trained
  network -- true inductive inference. Memorization baselines break here;
  only a model that generalizes over *structure* (a GNN) transfers.

Negatives are drawn from a hard-negative pool (same-module and cross-process
pairs), which is far more honest than uniform random negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.utils import subgraph

from .config import SplitConfig


@dataclass
class LinkSplit:
    """Edges + labels for one split, plus the graphs used for encoding."""

    train_edge_index: torch.Tensor   # message-passing graph during training
    eval_edge_index: torch.Tensor    # message-passing graph at eval time
    x: torch.Tensor                  # node features
    train_pairs: torch.Tensor        # (2, n_train)
    train_labels: torch.Tensor       # (n_train,)
    val_pairs: torch.Tensor
    val_labels: torch.Tensor
    test_pairs: torch.Tensor
    test_labels: torch.Tensor
    train_nodes: torch.Tensor        # node ids visible during training
    num_nodes: int
    # Optional relation ids per edge (for R-GCN); None for single-relation graphs.
    train_edge_type: torch.Tensor = None
    eval_edge_type: torch.Tensor = None
    num_relations: int = 1


def _split_indices(n: int, val_frac: float, test_frac: float,
                   rng: np.random.Generator):
    perm = rng.permutation(n)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    return perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]


def make_split(graph, cfg: SplitConfig, neg_ratio: int = 1) -> LinkSplit:
    """Build a train/val/test split according to ``cfg.mode``."""
    rng = np.random.default_rng(cfg.seed)
    if cfg.mode == "transductive":
        return _transductive_split(graph, cfg, neg_ratio, rng)
    if cfg.mode == "inductive":
        return _inductive_split(graph, cfg, neg_ratio, rng)
    raise ValueError(f"Unknown split mode: {cfg.mode}")


def _transductive_split(graph, cfg, neg_ratio, rng) -> LinkSplit:
    pos, neg = graph.sl_pos, graph.sl_neg_pool
    p_tr, p_va, p_te = _split_indices(pos.size(1), cfg.val_frac, cfg.test_frac, rng)
    n_tr, n_va, n_te = _split_indices(neg.size(1), cfg.val_frac, cfg.test_frac, rng)

    def combine(p_idx, n_idx):
        n_neg = min(len(n_idx), neg_ratio * len(p_idx))
        n_idx = n_idx[:n_neg]
        pairs = torch.cat([pos[:, p_idx], neg[:, n_idx]], dim=1)
        labels = torch.cat([torch.ones(len(p_idx)), torch.zeros(len(n_idx))])
        return pairs, labels

    tr_pairs, tr_labels = combine(p_tr, n_tr)
    va_pairs, va_labels = combine(p_va, n_va)
    te_pairs, te_labels = combine(p_te, n_te)
    ei = graph.data.edge_index
    et = getattr(graph.data, "edge_type", None)
    return LinkSplit(
        train_edge_index=ei, eval_edge_index=ei, x=graph.data.x,
        train_pairs=tr_pairs, train_labels=tr_labels,
        val_pairs=va_pairs, val_labels=va_labels,
        test_pairs=te_pairs, test_labels=te_labels,
        train_nodes=torch.arange(graph.num_nodes), num_nodes=graph.num_nodes,
        train_edge_type=et, eval_edge_type=et,
        num_relations=getattr(graph, "num_relations", 1),
    )


def _touch(pairs, mask):
    return mask[pairs[0]] | mask[pairs[1]]


def _within(pairs, mask):
    return mask[pairs[0]] & mask[pairs[1]]


def _inductive_split(graph, cfg, neg_ratio, rng) -> LinkSplit:
    pos, neg = graph.sl_pos, graph.sl_neg_pool
    num_nodes = graph.num_nodes
    perm = rng.permutation(num_nodes)
    n_test_nodes = int(cfg.test_frac * num_nodes)
    n_val_nodes = int(cfg.val_frac * num_nodes)
    test_nodes = perm[:n_test_nodes]
    val_nodes = perm[n_test_nodes:n_test_nodes + n_val_nodes]
    train_nodes = perm[n_test_nodes + n_val_nodes:]

    is_test = torch.zeros(num_nodes, dtype=torch.bool); is_test[test_nodes] = True
    is_val = torch.zeros(num_nodes, dtype=torch.bool); is_val[val_nodes] = True
    is_train = torch.zeros(num_nodes, dtype=torch.bool); is_train[train_nodes] = True

    def masks(pairs):
        test_m = _touch(pairs, is_test)
        val_m = _touch(pairs, is_val) & ~test_m
        train_m = _within(pairs, is_train)
        return train_m, val_m, test_m

    pos_tr, pos_va, pos_te = masks(pos)
    neg_tr, neg_va, neg_te = masks(neg)

    def combine(p_mask, n_mask):
        p = pos[:, p_mask]
        n_pool = neg[:, n_mask]
        n_neg = min(n_pool.size(1), neg_ratio * p.size(1))
        if n_pool.size(1) > 0 and n_neg > 0:
            idx = rng.choice(n_pool.size(1), size=n_neg, replace=False)
            n_sel = n_pool[:, idx]
        else:
            n_sel = torch.empty((2, 0), dtype=torch.long)
        pairs = torch.cat([p, n_sel], dim=1)
        labels = torch.cat([torch.ones(p.size(1)), torch.zeros(n_sel.size(1))])
        return pairs, labels

    tr_pairs, tr_labels = combine(pos_tr, neg_tr)
    va_pairs, va_labels = combine(pos_va, neg_va)
    te_pairs, te_labels = combine(pos_te, neg_te)

    train_node_tensor = torch.from_numpy(np.sort(train_nodes)).long()
    full_ei = graph.data.edge_index
    full_et = getattr(graph.data, "edge_type", None)
    train_ei, _, edge_mask = subgraph(
        train_node_tensor, full_ei, relabel_nodes=False,
        num_nodes=num_nodes, return_edge_mask=True)
    train_et = full_et[edge_mask] if full_et is not None else None
    return LinkSplit(
        train_edge_index=train_ei, eval_edge_index=full_ei, x=graph.data.x,
        train_pairs=tr_pairs, train_labels=tr_labels,
        val_pairs=va_pairs, val_labels=va_labels,
        test_pairs=te_pairs, test_labels=te_labels,
        train_nodes=train_node_tensor, num_nodes=num_nodes,
        train_edge_type=train_et, eval_edge_type=full_et,
        num_relations=getattr(graph, "num_relations", 1),
    )
