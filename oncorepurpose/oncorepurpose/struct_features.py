"""Non-semantic (structural) node features for the OncoEvidence feature ablation.

Motivation
----------
The text features (`data[nt].x` = SentenceTransformer embeddings of node *names*)
let a tabular model (XGBoost) exploit name similarity directly, with no graph
traversal.  To run the decisive "structure-as-features (XGBoost) vs
structure-via-message-passing (GNN)" test we replace every node's feature vector
with one that encodes *local graph structure only* -- no name/text semantics, and
the SAME construction for every node type:

    [ log1p(total_degree),
      log1p(per-relation incident-edge count) for the top-K global relations,
      one-hot(node_type) ]

The count columns are z-scored (per node type) so the linear projection / trees
see comparable magnitudes; the one-hot columns are left as raw 0/1.  Dimensionality
is identical across node types by construction (1 + K + n_node_types), so no
padding/truncation is needed.

Leakage control
---------------
Drug<->disease *therapeutic* relations (indication / contraindication / off-label,
both directions) are EXCLUDED from the degree/relation counts -- exactly the edges
the splitter strips from the message-passing graph.  Otherwise a held-out disease's
indication degree would directly count its own test positives.  Excluding them keeps
the structural features non-target (connectivity to proteins/pathways/phenotypes/...),
so the comparison stays fair.

`random` reference features (a fixed random vector per node) are also provided; they
cripple XGBoost to ~chance and isolate a pure-structure GNN (NOT a fair comparison,
reference only).
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, THERAPEUTIC_RELS
from oncorepurpose.data.build_graph import _norm_rel

_THERAPEUTIC_NORM = {_norm_rel(r) for r in THERAPEUTIC_RELS}


def _therapeutic_edge_types(data: HeteroData):
    """Drug<->disease therapeutic edge types (both directions) -- excluded from counts."""
    out = set()
    for et in data.edge_types:
        s, r, d = et
        if {s, d} == {DRUG_TYPE, DISEASE_TYPE} and r in _THERAPEUTIC_NORM:
            out.add(et)
    return out


def build_structural_features(
    data: HeteroData,
    top_k_relations: int = 12,
    standardize: bool = True,
    assign: bool = True,
) -> Dict[str, torch.Tensor]:
    """Replace each node type's `data[nt].x` with non-semantic structural features.

    Returns a dict {node_type: feature_tensor [num_nodes, D]} with identical D for
    every node type.  If `assign` is True, also writes the tensors into
    `data[nt].x` (in place) so downstream splits/models pick them up directly.
    """
    node_types: List[str] = list(data.node_types)
    type_index = {nt: i for i, nt in enumerate(node_types)}
    num_nodes = {nt: int(data[nt].num_nodes) for nt in node_types}

    therapeutic = _therapeutic_edge_types(data)

    # total degree per node (excluding therapeutic edges)
    total_deg = {nt: torch.zeros(num_nodes[nt], dtype=torch.float32) for nt in node_types}
    # per-relation incident counts: rel_counts[rel][nt] = tensor[num_nodes[nt]]
    rel_counts: Dict[str, Dict[str, torch.Tensor]] = {}
    rel_global_edges: Dict[str, int] = {}

    for et in data.edge_types:
        if et in therapeutic:
            continue
        s, r, d = et
        ei = data[et].edge_index
        if ei.numel() == 0:
            continue
        src_deg = torch.bincount(ei[0], minlength=num_nodes[s]).float()
        dst_deg = torch.bincount(ei[1], minlength=num_nodes[d]).float()
        total_deg[s] += src_deg
        total_deg[d] += dst_deg

        rel_global_edges[r] = rel_global_edges.get(r, 0) + int(ei.size(1))
        per = rel_counts.setdefault(r, {})
        if s not in per:
            per[s] = torch.zeros(num_nodes[s], dtype=torch.float32)
        if d not in per:
            per[d] = torch.zeros(num_nodes[d], dtype=torch.float32)
        per[s] += src_deg
        per[d] += dst_deg

    # most informative relations = top-K by global (non-therapeutic) edge count
    top_rels = sorted(rel_global_edges, key=rel_global_edges.get, reverse=True)[:top_k_relations]

    out: Dict[str, torch.Tensor] = {}
    for nt in node_types:
        n = num_nodes[nt]
        cols = [torch.log1p(total_deg[nt]).unsqueeze(1)]
        for r in top_rels:
            c = rel_counts.get(r, {}).get(nt)
            if c is None:
                c = torch.zeros(n, dtype=torch.float32)
            cols.append(torch.log1p(c).unsqueeze(1))
        counts = torch.cat(cols, dim=1)  # [n, 1+K]

        if standardize:
            mean = counts.mean(dim=0, keepdim=True)
            std = counts.std(dim=0, keepdim=True)
            counts = (counts - mean) / (std + 1e-6)

        onehot = torch.zeros(n, len(node_types), dtype=torch.float32)
        onehot[:, type_index[nt]] = 1.0

        x = torch.cat([counts, onehot], dim=1).float()
        out[nt] = x
        if assign:
            data[nt].x = x

    return out


def build_random_features(
    data: HeteroData, dim: int = 64, seed: int = 0, assign: bool = True
) -> Dict[str, torch.Tensor]:
    """Fixed random vector per node (REFERENCE ONLY -- cripples XGBoost to ~chance).

    Returns {node_type: tensor [num_nodes, dim]}; optionally assigns into data[nt].x.
    """
    gen = torch.Generator().manual_seed(seed)
    out: Dict[str, torch.Tensor] = {}
    for nt in data.node_types:
        n = int(data[nt].num_nodes)
        x = torch.randn(n, dim, generator=gen, dtype=torch.float32)
        out[nt] = x
        if assign:
            data[nt].x = x
    return out
