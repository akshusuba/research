"""Temporal (prospective) split for drug->disease link prediction on PrimeKG.

PrimeKG carries no edge timestamps, so the temporal axis is derived externally:
each true (drug, oncology-disease) indication pair is assigned an approximate
*first-evidence year* (earliest Europe PMC co-mention; see
``scripts/evaluate_temporal_split.py``). Given a cutoff year T:

- PAST pairs (year <= T) are treated as "known before the cutoff". A leakage-free
  subset of them seeds the message-passing graph and supplies train supervision.
- FUTURE pairs (year > T) are the held-out PROSPECTIVE test set: their target
  edges are REMOVED from the message-passing graph, so the model must rank them
  above sampled negatives using only PAST structure (could it have predicted the
  indication BEFORE it was established?).

This mirrors ``inductive_node_split`` in ``splits.py`` (same leakage controls:
restrict the target relation to train columns, strip ALL drug<->disease
therapeutic edges so a held-out pair is never revealed through a sibling
relation) but holds out edges by TIME rather than by node identity. It reuses
``_build_base_graph``, ``_sample_negatives`` and ``_make`` from ``splits.py`` and
returns the same ``SplitData`` object, so the unchanged trainer/models apply.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, ONCOLOGY_KEYWORDS
from oncorepurpose.evaluation.splits import (
    SplitData,
    _build_base_graph,
    _make,
    _sample_negatives,
    _therapeutic_edge_types,
)

EdgeType = Tuple[str, str, str]


def oncology_disease_set(data: HeteroData) -> Set[int]:
    """Indices of oncology diseases via the ``is_oncology`` mask, else keywords."""
    store = data[DISEASE_TYPE]
    if "is_oncology" in store:
        return set(torch.nonzero(store.is_oncology, as_tuple=False).flatten().tolist())
    names = list(getattr(store, "node_names", []) or [])
    out: Set[int] = set()
    for i, nm in enumerate(names):
        low = (nm or "").lower()
        if any(k in low for k in ONCOLOGY_KEYWORDS):
            out.add(i)
    return out


def _drug_disease_rows(target_edge_type: EdgeType) -> Tuple[int, int]:
    """Return (drug_row, disease_row) for the stored edge_index orientation."""
    s_t, _, d_t = target_edge_type
    if s_t == DRUG_TYPE and d_t == DISEASE_TYPE:
        return 0, 1
    if s_t == DISEASE_TYPE and d_t == DRUG_TYPE:
        return 1, 0
    raise ValueError(f"target_edge_type must connect drug and disease, got {target_edge_type}")


def true_oncology_pairs(data: HeteroData, target_edge_type: EdgeType) -> List[Tuple[int, int]]:
    """De-duplicated list of (drug_idx, disease_idx) for oncology indication edges."""
    ei = data[target_edge_type].edge_index
    drug_row, dis_row = _drug_disease_rows(target_edge_type)
    onco = oncology_disease_set(data)
    seen: Set[Tuple[int, int]] = set()
    pairs: List[Tuple[int, int]] = []
    for col in range(ei.size(1)):
        dr = int(ei[drug_row, col])
        ds = int(ei[dis_row, col])
        if ds in onco and (dr, ds) not in seen:
            seen.add((dr, ds))
            pairs.append((dr, ds))
    return pairs


def _known_therapeutic_pairs(data: HeteroData) -> Set[Tuple[int, int]]:
    """All (drug_idx, disease_idx) therapeutic pairs (both directions, all rels).

    Used to keep sampled negatives clean: a negative is never a real
    indication / contraindication / off-label pair.
    """
    known: Set[Tuple[int, int]] = set()
    for et in _therapeutic_edge_types(data):
        s, _, _ = et
        e = data[et].edge_index
        if s == DRUG_TYPE:
            for a, b in zip(e[0].tolist(), e[1].tolist()):
                known.add((a, b))
        else:
            for a, b in zip(e[0].tolist(), e[1].tolist()):
                known.add((b, a))
    return known


def temporal_split(
    data: HeteroData,
    target_edge_type: EdgeType,
    pair_years: Dict[Tuple[int, int], int],
    cutoff_year: int,
    onco_set: Optional[Set[int]] = None,
    neg_ratio: float = 1.0,
    test_neg_ratio: float = 5.0,
    val_frac: float = 0.15,
    seed: int = 0,
) -> SplitData:
    """Build a temporal SplitData.

    Parameters
    ----------
    pair_years : dict {(drug_idx, disease_idx): first-evidence year}
        Only pairs present here participate (those with a resolved year).
    cutoff_year : T. year <= T -> PAST (in graph / train); year > T -> FUTURE (test).
    onco_set : oncology disease indices for negative sampling (computed if None).
    neg_ratio : negatives per positive for train/val.
    test_neg_ratio : negatives per positive for the prospective test set
        (larger so AUPRC / recall@k reflect a realistic retrieval haystack).
    val_frac : fraction of PAST pairs held out (NOT in graph) for early stopping.
    """
    if onco_set is None:
        onco_set = oncology_disease_set(data)
    gen = torch.Generator().manual_seed(seed)
    s_t, _, d_t = target_edge_type
    drug_row, dis_row = _drug_disease_rows(target_edge_type)
    ei = data[target_edge_type].edge_index

    # Map each (drug, disease) pair to a representative target column.
    pair_to_col: Dict[Tuple[int, int], int] = {}
    for col in range(ei.size(1)):
        key = (int(ei[drug_row, col]), int(ei[dis_row, col]))
        pair_to_col.setdefault(key, col)

    past_cols: List[int] = []
    past_pairs: List[Tuple[int, int]] = []
    future_pairs: List[Tuple[int, int]] = []
    for (dr, ds), yr in pair_years.items():
        col = pair_to_col.get((dr, ds))
        if col is None:
            continue
        if yr <= cutoff_year:
            past_cols.append(col)
            past_pairs.append((dr, ds))
        else:
            future_pairs.append((dr, ds))

    # Carve a validation subset out of PAST (held out of message passing).
    n_past = len(past_pairs)
    perm = torch.randperm(n_past, generator=gen).tolist() if n_past else []
    n_val = int(round(val_frac * n_past)) if n_past > 1 else 0
    n_val = min(max(n_val, 1 if n_past > 1 else 0), max(n_past - 1, 0))
    val_local = set(perm[:n_val])
    train_local = [i for i in range(n_past) if i not in val_local]

    train_cols = torch.tensor([past_cols[i] for i in train_local], dtype=torch.long)

    # Message-passing graph: only PAST-train target edges; all therapeutic
    # drug<->disease edges stripped (so FUTURE / val pairs cannot leak).
    base = _build_base_graph(data, target_edge_type, train_cols, None, set())

    def to_index(pairs: List[Tuple[int, int]]) -> torch.Tensor:
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
        drugs = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dis = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        return torch.stack([drugs, dis]) if s_t == DRUG_TYPE else torch.stack([dis, drugs])

    train_pos = to_index([past_pairs[i] for i in train_local])
    val_pos = to_index([past_pairs[i] for i in sorted(val_local)])
    test_pos = to_index(future_pairs)

    # Negatives: random (drug, oncology disease) not in any therapeutic pair.
    known = _known_therapeutic_pairs(data)  # keyed (drug, disease)
    src_pool = torch.arange(int(data[DRUG_TYPE].num_nodes))
    dst_pool = torch.tensor(sorted(onco_set), dtype=torch.long)

    def neg(pos: torch.Tensor, ratio: float) -> torch.Tensor:
        num = int(round(pos.size(1) * ratio))
        di = _sample_negatives(known, src_pool, dst_pool, num, gen)  # (drug, disease)
        return di if s_t == DRUG_TYPE else torch.stack([di[1], di[0]])

    info = {
        "regime": "temporal",
        "cutoff_year": int(cutoff_year),
        "n_past_total": n_past,
        "n_train_pos": int(train_pos.size(1)),
        "n_val_pos": int(val_pos.size(1)),
        "n_future_pos": int(test_pos.size(1)),
        "neg_ratio": neg_ratio,
        "test_neg_ratio": test_neg_ratio,
    }
    return _make(
        base, target_edge_type, "temporal",
        train_pos, neg(train_pos, neg_ratio),
        val_pos, neg(val_pos, neg_ratio),
        test_pos, neg(test_pos, test_neg_ratio),
        info,
    )
