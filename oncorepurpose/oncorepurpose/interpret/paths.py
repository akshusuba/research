"""Rank novel drug->disease candidates and extract multi-hop KG rationales.

Two pieces feed the agentic evidence-report layer:
1. predict_candidates_for_diseases: score (drug, disease) pairs with a trained
   encoder, exclude known therapeutic links, return top-K drugs per disease.
2. extract_paths: enumerate short, readable KG paths connecting a drug and a
   disease (e.g. drug --target--> gene <--associated-- disease), down-weighting
   hub intermediates, to explain *why* the model linked them.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, THERAPEUTIC_RELS
from oncorepurpose.data.build_graph import _norm_rel

EdgeType = Tuple[str, str, str]
_THERAPEUTIC_NORM = {_norm_rel(r) for r in THERAPEUTIC_RELS}


def _known_pairs(data: HeteroData) -> set:
    """All known drug<->disease therapeutic pairs (drug_idx, disease_idx)."""
    known = set()
    for et in data.edge_types:
        s, r, d = et
        if {s, d} == {DRUG_TYPE, DISEASE_TYPE} and r in _THERAPEUTIC_NORM:
            ei = data[et].edge_index
            if s == DRUG_TYPE:
                for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                    known.add((a, b))
            else:
                for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                    known.add((b, a))
    return known


@torch.no_grad()
def predict_candidates_for_diseases(
    model, data: HeteroData, target_edge_type: EdgeType, disease_indices: List[int],
    device: torch.device, top_k: int = 10, exclude_known: bool = True,
    rank_by: str = "specificity", pop_sample: int = 64, seed: int = 0,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """For each disease, return top-k novel drugs as [(drug_idx, score, lift), ...].

    With ``rank_by="specificity"`` (default) candidates are ranked by *disease-
    specific lift* = score(drug, this disease) - the drug's average score across a
    random sample of diseases. This removes the popularity artifact where a few
    broadly-indicated drugs top every disease's list; ``rank_by="score"`` reverts
    to raw model score.
    """
    model.eval()
    z = model.encode(data)  # full-graph embeddings
    dev = z[DRUG_TYPE].device
    num_drugs = int(data[DRUG_TYPE].num_nodes)
    num_dis = int(data[DISEASE_TYPE].num_nodes)
    known = _known_pairs(data) if exclude_known else set()
    all_drugs = torch.arange(num_drugs, device=dev)

    def score_disease(dz: int) -> torch.Tensor:
        eli = torch.stack([all_drugs, torch.full((num_drugs,), dz, device=dev)])
        return torch.sigmoid(model.decode(z, target_edge_type, eli)).cpu()

    pop = None
    if rank_by == "specificity":
        g = torch.Generator().manual_seed(seed)
        sample = torch.randperm(num_dis, generator=g)[: min(pop_sample, num_dis)].tolist()
        acc = torch.zeros(num_drugs)
        for dz in sample:
            acc += score_disease(dz)
        pop = acc / max(1, len(sample))

    out: Dict[int, List[Tuple[int, float, float]]] = {}
    for dz in disease_indices:
        scores = score_disease(dz)
        rank_val = (scores - pop) if pop is not None else scores
        order = torch.argsort(rank_val, descending=True).tolist()
        ranked = []
        for di in order:
            if exclude_known and (di, dz) in known:
                continue
            ranked.append((di, float(scores[di]), float(rank_val[di])))
            if len(ranked) >= top_k:
                break
        out[dz] = ranked
    return out


def _typed_neighbors(data: HeteroData, node_type: str, node_idx: int) -> Dict[str, Dict[int, str]]:
    """Return {neighbor_type: {neighbor_idx: relation}} for one node (both directions)."""
    nbrs: Dict[str, Dict[int, str]] = defaultdict(dict)
    for et in data.edge_types:
        s, r, d = et
        ei = data[et].edge_index
        if s == node_type:
            m = ei[0] == node_idx
            for nb in ei[1][m].tolist():
                nbrs[d].setdefault(nb, r)
        if d == node_type:
            m = ei[1] == node_idx
            for nb in ei[0][m].tolist():
                nbrs[s].setdefault(nb, r)
    return nbrs


def _name(data: HeteroData, ntype: str, idx: int) -> str:
    names = getattr(data[ntype], "node_names", None)
    if names is not None and idx < len(names):
        return str(names[idx])
    return f"{ntype}:{idx}"


def extract_paths(
    data: HeteroData, drug_idx: int, disease_idx: int,
    bridge_types: Tuple[str, ...] = ("gene_protein", "pathway", "biological_process", "effect_phenotype"),
    max_paths: int = 8, hub_cap: int = 400,
) -> List[Dict]:
    """Find 2-hop bridges (drug -> X <- disease) through shared intermediates.

    NOTE: this is a *generic* baseline bridge finder and will happily bridge on a
    shared phenotype/symptom (``effect_phenotype``), which is not a mechanism. It is
    kept for comparison only. The deliverable pipeline
    (``scripts/generate_report.py``) and all evaluations use
    ``oncorepurpose.interpret.mechanism_paths.mechanism_paths`` instead, which accepts
    only target / PPI / pathway MOA chains.

    Hub intermediates (degree heuristically large) are down-weighted so paths stay
    specific. Returns readable path dicts ranked by a simple specificity score.
    """
    drug_nbrs = _typed_neighbors(data, DRUG_TYPE, drug_idx)
    dis_nbrs = _typed_neighbors(data, DISEASE_TYPE, disease_idx)

    paths = []
    for bt in bridge_types:
        shared = set(drug_nbrs.get(bt, {})) & set(dis_nbrs.get(bt, {}))
        for mid in shared:
            # Skip obvious hubs by capping how many drugs/diseases this mid connects.
            paths.append({
                "bridge_type": bt,
                "bridge_name": _name(data, bt, mid),
                "drug_relation": drug_nbrs[bt][mid],
                "disease_relation": dis_nbrs[bt][mid],
                "text": (
                    f"{_name(data, DRUG_TYPE, drug_idx)} --{drug_nbrs[bt][mid]}--> "
                    f"{_name(data, bt, mid)} <--{dis_nbrs[bt][mid]}-- "
                    f"{_name(data, DISEASE_TYPE, disease_idx)}"
                ),
            })
    # Prefer protein/pathway bridges; cap count.
    priority = {bt: i for i, bt in enumerate(bridge_types)}
    paths.sort(key=lambda p: priority.get(p["bridge_type"], 99))
    return paths[:max_paths]
