"""Load the PrimeKG HeteroData with shared features and resolve target edges."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import (
    CONTRAINDICATION_REL,
    DISEASE_TYPE,
    DRUG_TYPE,
    HETERODATA_PT,
    INDICATION_REL,
)
from oncorepurpose.data.build_graph import _norm_rel, build_hetero_from_primekg
from oncorepurpose.features import build_text_features

EdgeType = Tuple[str, str, str]


def _find_target_edge(data: HeteroData, relation: str) -> Optional[EdgeType]:
    """Find the (drug, <relation>, disease) edge type, tolerant to normalization."""
    rel_n = _norm_rel(relation)
    candidates = []
    for et in data.edge_types:
        s, r, d = et
        if {s, d} == {DRUG_TYPE, DISEASE_TYPE} and r == rel_n:
            candidates.append(et)
    if not candidates:
        return None
    # Prefer drug->disease direction.
    for et in candidates:
        if et[0] == DRUG_TYPE:
            return et
    return candidates[0]


def load_primekg(
    pt_path: Path = HETERODATA_PT,
    with_features: bool = True,
    force_fallback_features: bool = False,
    build_if_missing: bool = True,
) -> Tuple[HeteroData, dict]:
    """Load PrimeKG HeteroData + attach features + resolve target edge types.

    Returns (data, targets) where targets = {"indication": EdgeType, "contraindication": EdgeType}.
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"{pt_path} not found; build the graph first.")
        data = build_hetero_from_primekg(save=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = torch.load(pt_path, weights_only=False)

    for nt in data.node_types:
        if not hasattr(data[nt], "num_nodes") or data[nt].num_nodes is None:
            names = getattr(data[nt], "node_names", None)
            data[nt].num_nodes = len(names) if names is not None else 0

    if with_features:
        build_text_features(data, force_fallback=force_fallback_features)

    targets = {
        "indication": _find_target_edge(data, INDICATION_REL),
        "contraindication": _find_target_edge(data, CONTRAINDICATION_REL),
    }
    return data, targets


def graph_summary(data: HeteroData, targets: dict) -> str:
    lines = ["PrimeKG graph:"]
    total_nodes = total_edges = 0
    for nt in data.node_types:
        n = int(data[nt].num_nodes)
        total_nodes += n
        dim = int(data[nt].x.size(1)) if "x" in data[nt] else None
        extra = f" feat={dim}" if dim else ""
        if nt == DISEASE_TYPE and "is_oncology" in data[nt]:
            extra += f" oncology={int(data[nt].is_oncology.sum())}"
        lines.append(f"  node {nt}: {n}{extra}")
    for et in data.edge_types:
        total_edges += int(data[et].edge_index.size(1))
    lines.append(f"  totals: {total_nodes:,} nodes, {total_edges:,} edges, {len(data.edge_types)} edge types")
    for name, et in targets.items():
        if et is not None:
            lines.append(f"  target [{name}]: {et} = {int(data[et].edge_index.size(1))} edges")
        else:
            lines.append(f"  target [{name}]: NOT FOUND")
    return "\n".join(lines)
