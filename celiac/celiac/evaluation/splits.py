"""
Train/val/test splits for link prediction.

Two regimes, returned in an identical format so the same training loop and the
same models can be used for both:

- ``transductive_split``: random edge holdout. Every node is seen during
  training; the task is to reconnect known nodes. This is the regime in which a
  memorising embedding-lookup model is hard to beat.

- ``inductive_node_split``: node-disjoint holdout. A fraction of nodes on one
  side of the target relation are held out entirely -- *none* of their target
  edges appear in training. At test time the model must score links for nodes it
  never trained on. Embedding-lookup models have no trained embedding for these
  nodes and collapse toward chance, whereas a feature-based GNN can still build a
  representation from the node's content features and its (non-target)
  neighbourhood. This is the regime that reveals whether the GNN adds value.

Output contract (both regimes), for ``target_edge_type``:
- All three returned ``HeteroData`` objects share the SAME message-passing graph
  (training edges only -- no leakage of val/test target edges).
- ``data[target_edge_type].edge_index`` holds the message-passing (train) target
  edges.
- ``data[target_edge_type].edge_label_index`` / ``.edge_label`` hold that split's
  supervision (positives + sampled negatives).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData

EdgeType = Tuple[str, str, str]


@dataclass
class SplitData:
    """A train/val/test split for one target edge type."""

    train: HeteroData
    val: HeteroData
    test: HeteroData
    target_edge_type: EdgeType
    regime: str
    info: dict


def _reverse_edge_type(et: EdgeType) -> EdgeType:
    src, rel, dst = et
    return (dst, rel, src)


def _base_graph(data: HeteroData, target_edge_type: EdgeType, train_pos: torch.Tensor) -> HeteroData:
    """Clone the graph, set target message edges to train positives, drop reverse target."""
    base = data.clone()
    base[target_edge_type].edge_index = train_pos.clone()
    # Remove any attributes on the target store that could leak full supervision.
    for attr in ("edge_label", "edge_label_index", "edge_weight"):
        if attr in base[target_edge_type]:
            del base[target_edge_type][attr]
    # Drop the (small) reverse relation so val/test target links cannot leak
    # through it during message passing. Treated identically in both regimes.
    rev = _reverse_edge_type(target_edge_type)
    if rev in base.edge_types and rev != target_edge_type:
        del base[rev]
    return base


def _sample_negatives(
    pos_set: Set[Tuple[int, int]],
    src_pool: torch.Tensor,
    dst_pool: torch.Tensor,
    num: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``num`` negative (src, dst) pairs not present in ``pos_set``."""
    if num <= 0 or src_pool.numel() == 0 or dst_pool.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    neg_src: List[int] = []
    neg_dst: List[int] = []
    seen: Set[Tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(1000, num * 50)

    while len(neg_src) < num and attempts < max_attempts:
        batch = max(num * 2, 64)
        si = torch.randint(0, src_pool.numel(), (batch,), generator=generator)
        di = torch.randint(0, dst_pool.numel(), (batch,), generator=generator)
        for s_idx, d_idx in zip(si.tolist(), di.tolist()):
            s = int(src_pool[s_idx])
            d = int(dst_pool[d_idx])
            key = (s, d)
            if key in pos_set or key in seen:
                continue
            seen.add(key)
            neg_src.append(s)
            neg_dst.append(d)
            if len(neg_src) >= num:
                break
        attempts += batch

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def _attach_supervision(
    base: HeteroData,
    target_edge_type: EdgeType,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> HeteroData:
    """Clone ``base`` and attach (pos + neg) supervision for the target edge."""
    d = base.clone()
    edge_label_index = torch.cat([pos, neg], dim=1)
    edge_label = torch.cat([
        torch.ones(pos.size(1), dtype=torch.float),
        torch.zeros(neg.size(1), dtype=torch.float),
    ])
    d[target_edge_type].edge_label_index = edge_label_index
    d[target_edge_type].edge_label = edge_label
    return d


def transductive_split(
    data: HeteroData,
    target_edge_type: EdgeType,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    neg_ratio: float = 1.0,
    seed: int = 0,
) -> SplitData:
    """Random edge holdout. Every node is seen during training."""
    g = torch.Generator().manual_seed(seed)

    pos_all = data[target_edge_type].edge_index
    num_edges = pos_all.size(1)
    perm = torch.randperm(num_edges, generator=g)

    n_test = int(round(test_frac * num_edges))
    n_val = int(round(val_frac * num_edges))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    train_pos = pos_all[:, train_idx]
    val_pos = pos_all[:, val_idx]
    test_pos = pos_all[:, test_idx]

    src_type, _, dst_type = target_edge_type
    src_pool = torch.arange(int(data[src_type].num_nodes))
    dst_pool = torch.arange(int(data[dst_type].num_nodes))
    pos_set = {(int(s), int(d)) for s, d in zip(pos_all[0].tolist(), pos_all[1].tolist())}

    base = _base_graph(data, target_edge_type, train_pos)

    def neg_for(pos: torch.Tensor) -> torch.Tensor:
        return _sample_negatives(pos_set, src_pool, dst_pool, int(round(pos.size(1) * neg_ratio)), g)

    train = _attach_supervision(base, target_edge_type, train_pos, neg_for(train_pos))
    val = _attach_supervision(base, target_edge_type, val_pos, neg_for(val_pos))
    test = _attach_supervision(base, target_edge_type, test_pos, neg_for(test_pos))

    info = {
        "regime": "transductive",
        "train_pos": int(train_pos.size(1)),
        "val_pos": int(val_pos.size(1)),
        "test_pos": int(test_pos.size(1)),
    }
    return SplitData(train, val, test, target_edge_type, "transductive", info)


def inductive_node_split(
    data: HeteroData,
    target_edge_type: EdgeType,
    holdout_side: str = "src",
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    neg_ratio: float = 1.0,
    seed: int = 0,
) -> SplitData:
    """Node-disjoint holdout.

    A fraction of nodes on ``holdout_side`` ("src" or "dst") are held out: ALL of
    their target edges are removed from training and routed to val/test. Their
    non-target edges remain in the graph, so a feature-based GNN can still build a
    representation from the surrounding structure -- but no model ever trains on
    their target links.
    """
    if holdout_side not in ("src", "dst"):
        raise ValueError("holdout_side must be 'src' or 'dst'")

    g = torch.Generator().manual_seed(seed)
    src_type, _, dst_type = target_edge_type
    pos_all = data[target_edge_type].edge_index

    side_type = src_type if holdout_side == "src" else dst_type
    side_row = 0 if holdout_side == "src" else 1

    # Only nodes that actually participate in target edges are candidates.
    participating = torch.unique(pos_all[side_row])
    perm = participating[torch.randperm(participating.numel(), generator=g)]

    n_nodes = participating.numel()
    n_test = max(1, int(round(test_frac * n_nodes)))
    n_val = max(1, int(round(val_frac * n_nodes)))
    test_nodes = set(perm[:n_test].tolist())
    val_nodes = set(perm[n_test:n_test + n_val].tolist())

    side_idx = pos_all[side_row].tolist()
    train_cols, val_cols, test_cols = [], [], []
    for col, node in enumerate(side_idx):
        if node in test_nodes:
            test_cols.append(col)
        elif node in val_nodes:
            val_cols.append(col)
        else:
            train_cols.append(col)

    train_pos = pos_all[:, torch.tensor(train_cols, dtype=torch.long)]
    val_pos = pos_all[:, torch.tensor(val_cols, dtype=torch.long)] if val_cols else pos_all[:, :0]
    test_pos = pos_all[:, torch.tensor(test_cols, dtype=torch.long)] if test_cols else pos_all[:, :0]

    pos_set = {(int(s), int(d)) for s, d in zip(pos_all[0].tolist(), pos_all[1].tolist())}

    other_type = dst_type if holdout_side == "src" else src_type
    other_pool = torch.arange(int(data[other_type].num_nodes))

    def held_pool(node_set: Set[int]) -> torch.Tensor:
        return torch.tensor(sorted(node_set), dtype=torch.long)

    train_side_nodes = torch.tensor(
        sorted(set(participating.tolist()) - test_nodes - val_nodes), dtype=torch.long
    )

    base = _base_graph(data, target_edge_type, train_pos)

    def neg_for(pos: torch.Tensor, side_pool: torch.Tensor) -> torch.Tensor:
        num = int(round(pos.size(1) * neg_ratio))
        if holdout_side == "src":
            return _sample_negatives(pos_set, side_pool, other_pool, num, g)
        return _sample_negatives(pos_set, other_pool, side_pool, num, g)

    train = _attach_supervision(base, target_edge_type, train_pos, neg_for(train_pos, train_side_nodes))
    val = _attach_supervision(base, target_edge_type, val_pos, neg_for(val_pos, held_pool(val_nodes)))
    test = _attach_supervision(base, target_edge_type, test_pos, neg_for(test_pos, held_pool(test_nodes)))

    info = {
        "regime": "inductive",
        "holdout_side": holdout_side,
        "held_side_type": side_type,
        "n_test_nodes": len(test_nodes),
        "n_val_nodes": len(val_nodes),
        "train_pos": int(train_pos.size(1)),
        "val_pos": int(val_pos.size(1)),
        "test_pos": int(test_pos.size(1)),
    }
    return SplitData(train, val, test, target_edge_type, "inductive", info)


def make_split(
    data: HeteroData,
    target_edge_type: EdgeType,
    regime: str,
    seed: int = 0,
    holdout_side: str = "src",
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    neg_ratio: float = 1.0,
) -> SplitData:
    """Dispatch to the requested split regime."""
    if regime == "transductive":
        return transductive_split(data, target_edge_type, val_frac, test_frac, neg_ratio, seed)
    if regime == "inductive":
        return inductive_node_split(
            data, target_edge_type, holdout_side, val_frac, test_frac, neg_ratio, seed
        )
    raise ValueError(f"Unknown regime: {regime}")
