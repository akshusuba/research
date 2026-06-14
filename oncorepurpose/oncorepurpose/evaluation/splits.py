"""Leakage-safe train/val/test splits for drug->disease link prediction on PrimeKG.

Three regimes, all returning the SAME message-passing graph (training edges only)
plus per-split supervision (positives + sampled negatives), so the identical
trainer and models serve every regime:

- transductive: random edge holdout; every node seen in training.
- inductive cold-disease: hold out whole diseases (optionally restricted to the
  oncology set). ALL of a held-out disease's drug-therapeutic edges (indication,
  contraindication, off-label, both directions) are removed from the training
  message-passing graph; its structural edges (disease-protein/phenotype/...) are
  kept, so a feature GNN can still embed it but never trained on its drug links.
- inductive cold-drug: same, holding out whole drugs.

Memory note: PrimeKG has 8.1M edges, so we build ONE shared base graph and return
lightweight supervision tensors rather than cloning the graph per split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, THERAPEUTIC_RELS
from oncorepurpose.data.build_graph import _norm_rel

EdgeType = Tuple[str, str, str]
_THERAPEUTIC_NORM = {_norm_rel(r) for r in THERAPEUTIC_RELS}


@dataclass
class SplitData:
    base: HeteroData                      # shared message-passing graph (train edges)
    target_edge_type: EdgeType
    regime: str
    train_label_index: torch.Tensor       # [2, P+N]
    train_label: torch.Tensor             # [P+N]
    val_label_index: torch.Tensor
    val_label: torch.Tensor
    test_label_index: torch.Tensor
    test_label: torch.Tensor
    info: dict = field(default_factory=dict)


def _therapeutic_edge_types(data: HeteroData) -> List[EdgeType]:
    """All drug<->disease therapeutic edge types (both directions)."""
    out = []
    for et in data.edge_types:
        s, r, d = et
        if {s, d} == {DRUG_TYPE, DISEASE_TYPE} and r in _THERAPEUTIC_NORM:
            out.append(et)
    return out


def _sample_negatives(
    pos_set: Set[Tuple[int, int]],
    src_pool: torch.Tensor,
    dst_pool: torch.Tensor,
    num: int,
    gen: torch.Generator,
) -> torch.Tensor:
    if num <= 0 or src_pool.numel() == 0 or dst_pool.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    ns, nd, seen = [], [], set()
    attempts, max_attempts = 0, max(2000, num * 50)
    while len(ns) < num and attempts < max_attempts:
        batch = max(num * 2, 128)
        si = src_pool[torch.randint(0, src_pool.numel(), (batch,), generator=gen)]
        di = dst_pool[torch.randint(0, dst_pool.numel(), (batch,), generator=gen)]
        for s, d in zip(si.tolist(), di.tolist()):
            key = (s, d)
            if key in pos_set or key in seen:
                continue
            seen.add(key); ns.append(s); nd.append(d)
            if len(ns) >= num:
                break
        attempts += batch
    return torch.tensor([ns, nd], dtype=torch.long)


def _build_base_graph(
    data: HeteroData,
    target_edge_type: EdgeType,
    train_target_cols: torch.Tensor,
    drop_node_side: Optional[str],
    held_nodes: Set[int],
) -> HeteroData:
    """Clone graph topology (not features-heavy deep copy) into a base message graph.

    We construct a fresh HeteroData sharing node feature tensors (cheap reference)
    and copying edge_index tensors, then (a) restrict the target relation to train
    columns and (b) strip therapeutic edges incident to held-out nodes.
    """
    base = HeteroData()
    for nt in data.node_types:
        for key, val in data[nt].items():
            base[nt][key] = val  # share references (x, node_names, num_nodes, masks)

    therapeutic = set(_therapeutic_edge_types(data))
    s_t, _, d_t = target_edge_type

    for et in data.edge_types:
        ei = data[et].edge_index
        if et == target_edge_type:
            ei = ei[:, train_target_cols]
        elif et in therapeutic and held_nodes:
            # Remove therapeutic edges touching held-out nodes (both directions).
            s_type, _, d_type = et
            keep = torch.ones(ei.size(1), dtype=torch.bool)
            if drop_node_side == "src":
                if s_type == s_t:
                    keep &= ~torch.tensor([n in held_nodes for n in ei[0].tolist()])
                if d_type == s_t:
                    keep &= ~torch.tensor([n in held_nodes for n in ei[1].tolist()])
            else:  # held nodes are disease side
                if s_type == d_t:
                    keep &= ~torch.tensor([n in held_nodes for n in ei[0].tolist()])
                if d_type == d_t:
                    keep &= ~torch.tensor([n in held_nodes for n in ei[1].tolist()])
            ei = ei[:, keep]
        base[et].edge_index = ei
    return base


def _make(
    base: HeteroData, target: EdgeType, regime: str,
    tr_pos, tr_neg, va_pos, va_neg, te_pos, te_neg, info,
) -> SplitData:
    def cat(pos, neg):
        eli = torch.cat([pos, neg], dim=1)
        lab = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
        return eli, lab
    tr_i, tr_l = cat(tr_pos, tr_neg)
    va_i, va_l = cat(va_pos, va_neg)
    te_i, te_l = cat(te_pos, te_neg)
    return SplitData(base, target, regime, tr_i, tr_l, va_i, va_l, te_i, te_l, info)


def transductive_split(
    data: HeteroData, target_edge_type: EdgeType,
    val_frac: float = 0.1, test_frac: float = 0.2, neg_ratio: float = 1.0, seed: int = 0,
) -> SplitData:
    gen = torch.Generator().manual_seed(seed)
    s_t, _, d_t = target_edge_type
    pos = data[target_edge_type].edge_index
    n = pos.size(1)
    perm = torch.randperm(n, generator=gen)
    n_test, n_val = int(round(test_frac * n)), int(round(val_frac * n))
    test_c, val_c, train_c = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    tr_pos, va_pos, te_pos = pos[:, train_c], pos[:, val_c], pos[:, test_c]

    pos_set = {(int(s), int(d)) for s, d in zip(pos[0].tolist(), pos[1].tolist())}
    src_pool = torch.arange(int(data[s_t].num_nodes))
    dst_pool = torch.arange(int(data[d_t].num_nodes))
    base = _build_base_graph(data, target_edge_type, train_c, None, set())

    def neg(p):
        return _sample_negatives(pos_set, src_pool, dst_pool, int(round(p.size(1) * neg_ratio)), gen)
    info = {"regime": "transductive", "train_pos": int(tr_pos.size(1)),
            "val_pos": int(va_pos.size(1)), "test_pos": int(te_pos.size(1))}
    return _make(base, target_edge_type, "transductive",
                 tr_pos, neg(tr_pos), va_pos, neg(va_pos), te_pos, neg(te_pos), info)


def inductive_node_split(
    data: HeteroData, target_edge_type: EdgeType, holdout_side: str = "dst",
    val_frac: float = 0.1, test_frac: float = 0.2, neg_ratio: float = 1.0, seed: int = 0,
    restrict_oncology: bool = False,
) -> SplitData:
    if holdout_side not in ("src", "dst"):
        raise ValueError("holdout_side must be 'src' or 'dst'")
    gen = torch.Generator().manual_seed(seed)
    s_t, _, d_t = target_edge_type
    pos = data[target_edge_type].edge_index
    side_row = 0 if holdout_side == "src" else 1
    side_type = s_t if holdout_side == "src" else d_t

    participating = torch.unique(pos[side_row])
    if restrict_oncology and holdout_side == "dst" and "is_oncology" in data[d_t]:
        onc = data[d_t].is_oncology
        participating = torch.tensor([n for n in participating.tolist() if bool(onc[n])], dtype=torch.long)

    perm = participating[torch.randperm(participating.numel(), generator=gen)]
    n = participating.numel()
    n_test, n_val = max(1, int(round(test_frac * n))), max(1, int(round(val_frac * n)))
    test_nodes = set(perm[:n_test].tolist())
    val_nodes = set(perm[n_test:n_test + n_val].tolist())
    held = test_nodes | val_nodes

    side_idx = pos[side_row].tolist()
    tr_c, va_c, te_c = [], [], []
    for col, nd in enumerate(side_idx):
        if nd in test_nodes:
            te_c.append(col)
        elif nd in val_nodes:
            va_c.append(col)
        else:
            tr_c.append(col)
    tr_c = torch.tensor(tr_c, dtype=torch.long)
    tr_pos = pos[:, tr_c]
    va_pos = pos[:, torch.tensor(va_c, dtype=torch.long)] if va_c else pos[:, :0]
    te_pos = pos[:, torch.tensor(te_c, dtype=torch.long)] if te_c else pos[:, :0]

    pos_set = {(int(s), int(d)) for s, d in zip(pos[0].tolist(), pos[1].tolist())}
    other_type = d_t if holdout_side == "src" else s_t
    other_pool = torch.arange(int(data[other_type].num_nodes))
    train_side = torch.tensor(sorted(set(participating.tolist()) - held), dtype=torch.long)

    base = _build_base_graph(data, target_edge_type, tr_c, holdout_side, held)

    def neg(p, side_pool):
        num = int(round(p.size(1) * neg_ratio))
        if holdout_side == "src":
            return _sample_negatives(pos_set, side_pool, other_pool, num, gen)
        return _sample_negatives(pos_set, other_pool, side_pool, num, gen)

    def pool(s):
        return torch.tensor(sorted(s), dtype=torch.long)

    info = {"regime": f"inductive_cold_{holdout_side}", "held_side_type": side_type,
            "restrict_oncology": restrict_oncology, "n_test_nodes": len(test_nodes),
            "n_val_nodes": len(val_nodes), "train_pos": int(tr_pos.size(1)),
            "val_pos": int(va_pos.size(1)), "test_pos": int(te_pos.size(1))}
    return _make(base, target_edge_type, info["regime"],
                 tr_pos, neg(tr_pos, train_side), va_pos, neg(va_pos, pool(val_nodes)),
                 te_pos, neg(te_pos, pool(test_nodes)), info)


def ablate_topology(split: SplitData, mode: str, seed: int = 0) -> SplitData:
    """Return a copy of `split` with its message-passing topology ablated.

    - 'empty': remove all non-target message edges (GNN -> feature-only, ~MLP).
    - 'shuffle': randomly permute each edge type's destination endpoints
      (destroys real connectivity while preserving per-type edge counts/degrees-in).
    The target relation's train edges are kept either way (they are supervision-
    derived, not "extra" structure), so only auxiliary topology is ablated.
    """
    gen = torch.Generator().manual_seed(seed)
    new_base = HeteroData()
    for nt in split.base.node_types:
        for key, val in split.base[nt].items():
            new_base[nt][key] = val
    for et in split.base.edge_types:
        ei = split.base[et].edge_index
        if et == split.target_edge_type:
            new_base[et].edge_index = ei
            continue
        if mode == "empty":
            new_base[et].edge_index = ei[:, :0]
        elif mode == "shuffle":
            perm = torch.randperm(ei.size(1), generator=gen)
            new_base[et].edge_index = torch.stack([ei[0], ei[1][perm]], dim=0)
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")
    return SplitData(new_base, split.target_edge_type, f"{split.regime}_ablate_{mode}",
                     split.train_label_index, split.train_label, split.val_label_index,
                     split.val_label, split.test_label_index, split.test_label, dict(split.info))


def drop_relations(split: SplitData, drop_substrings: List[str]) -> SplitData:
    """Return a copy of `split` with edge types whose relation matches any
    substring removed from the message-passing graph (relation ablation)."""
    new_base = HeteroData()
    for nt in split.base.node_types:
        for key, val in split.base[nt].items():
            new_base[nt][key] = val
    dropped = []
    for et in split.base.edge_types:
        _, rel, _ = et
        if et != split.target_edge_type and any(sub in rel for sub in drop_substrings):
            dropped.append(et)
            continue
        new_base[et].edge_index = split.base[et].edge_index
    info = dict(split.info)
    info["dropped_relations"] = [str(e) for e in dropped]
    return SplitData(new_base, split.target_edge_type, f"{split.regime}_drop",
                     split.train_label_index, split.train_label, split.val_label_index,
                     split.val_label, split.test_label_index, split.test_label, info)


def make_split(
    data: HeteroData, target_edge_type: EdgeType, regime: str, seed: int = 0,
    holdout_side: str = "dst", val_frac: float = 0.1, test_frac: float = 0.2,
    neg_ratio: float = 1.0, restrict_oncology: bool = False,
) -> SplitData:
    if regime == "transductive":
        return transductive_split(data, target_edge_type, val_frac, test_frac, neg_ratio, seed)
    if regime in ("inductive", "inductive_cold_dst", "inductive_cold_src"):
        side = "src" if regime.endswith("src") else holdout_side
        return inductive_node_split(data, target_edge_type, side, val_frac, test_frac,
                                    neg_ratio, seed, restrict_oncology)
    raise ValueError(f"Unknown regime: {regime}")
