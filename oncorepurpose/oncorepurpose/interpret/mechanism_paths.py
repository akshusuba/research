"""Multi-hop mechanism-of-action (MOA) path extraction over PrimeKG.

This is the graph's *real job* in OncoEvidence. A tuned tabular model can rank a
drug-disease pair, but only the knowledge graph can produce a traceable
mechanistic chain. We connect a drug to a disease through *mechanism* relations
(drug -> target protein, protein-protein interaction, protein -> pathway,
protein -> disease), as opposed to coincidental phenotype/symptom overlap
(e.g. a drug whose side effect happens to be a symptom of the disease).

Three canonical templates, in decreasing specificity:

  direct_target   drug -[targets]-> P            <-[assoc]- disease
  ppi             drug -[targets]-> P1 -[PPI]-> P2 <-[assoc]- disease
  shared_pathway  drug -[targets]-> P1 -[in]-> pathway <-[in]- P2 <-[assoc]- disease

Hub intermediates (proteins linked to very many diseases, or huge generic
pathways) are filtered/down-weighted so the paths stay specific and mechanistic.
Phenotype/symptom relations (`drug_effect`, `disease_phenotype_*`) and the
massive `drug_drug` / `anatomy_protein` edges are deliberately excluded.
"""

from __future__ import annotations

from collections import defaultdict
from math import log2
from typing import Dict, List

from torch_geometric.data import HeteroData

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE

PROT = "gene_protein"
PATHWAY = "pathway"


def _edge_pairs(data: HeteroData, et):
    ei = data[et].edge_index
    return ei[0].tolist(), ei[1].tolist()


def build_mech_index(data: HeteroData) -> dict:
    """Precompute the mechanism adjacency maps once (reusable across candidates)."""
    idx: Dict[str, defaultdict] = {
        "drug2prot": defaultdict(set), "dis2prot": defaultdict(set),
        "prot2dis": defaultdict(set), "ppi": defaultdict(set),
        "prot2pw": defaultdict(set), "pw2prot": defaultdict(set),
    }
    ets = set(data.edge_types)

    et = (DRUG_TYPE, "drug_protein", PROT)
    if et in ets:
        s, d = _edge_pairs(data, et)
        for a, b in zip(s, d):
            idx["drug2prot"][a].add(b)

    et = (DISEASE_TYPE, "disease_protein", PROT)
    if et in ets:
        s, d = _edge_pairs(data, et)
        for a, b in zip(s, d):
            idx["dis2prot"][a].add(b)
            idx["prot2dis"][b].add(a)

    et = (PROT, "protein_protein", PROT)
    if et in ets:
        s, d = _edge_pairs(data, et)
        for a, b in zip(s, d):
            idx["ppi"][a].add(b)
            idx["ppi"][b].add(a)

    et = (PROT, "pathway_protein", PATHWAY)
    if et in ets:
        s, d = _edge_pairs(data, et)
        for a, b in zip(s, d):
            idx["prot2pw"][a].add(b)
            idx["pw2prot"][b].add(a)

    idx["prot_dis_deg"] = {p: len(v) for p, v in idx["prot2dis"].items()}
    idx["pw_size"] = {pw: len(v) for pw, v in idx["pw2prot"].items()}
    return idx


def _name(data: HeteroData, nt: str, i: int) -> str:
    names = getattr(data[nt], "node_names", None)
    return str(names[i]) if names is not None and i < len(names) else f"{nt}:{i}"


def _spec(deg: int) -> float:
    """Specificity weight: a rarer intermediate (lower degree) scores higher."""
    return 1.0 / log2(deg + 2)


def mechanism_paths(
    data: HeteroData, idx: dict, drug_idx: int, disease_idx: int,
    max_paths: int = 8, prot_dis_cap: int = 150, pw_size_cap: int = 300,
    max_targets: int = 40,
) -> List[Dict]:
    """Return ranked MOA paths connecting `drug_idx` to `disease_idx`.

    Each path dict has: type, len, score, bridge proteins/pathway, and `text`.
    """
    targets = list(idx["drug2prot"].get(drug_idx, set()))[:max_targets]
    dis_prots = idx["dis2prot"].get(disease_idx, set())
    drug_n = _name(data, DRUG_TYPE, drug_idx)
    dis_n = _name(data, DISEASE_TYPE, disease_idx)
    out: List[Dict] = []

    # 1) direct target: the drug's target IS a disease-associated protein.
    for p in targets:
        if p in dis_prots:
            deg = idx["prot_dis_deg"].get(p, 1)
            if deg > prot_dis_cap:
                continue
            pn = _name(data, PROT, p)
            out.append({
                "type": "direct_target", "len": 2, "score": 3.0 + _spec(deg),
                "drug": drug_n, "disease": dis_n, "genes": [pn], "pathway": None,
                "text": f"{drug_n} --targets--> {pn} <--associated-- {dis_n}",
            })

    # 2) PPI bridge: drug target interacts with a disease protein.
    for p1 in targets:
        p1n = _name(data, PROT, p1)
        for p2 in idx["ppi"].get(p1, set()) & dis_prots:
            deg = idx["prot_dis_deg"].get(p2, 1)
            if deg > prot_dis_cap:
                continue
            p2n = _name(data, PROT, p2)
            out.append({
                "type": "ppi", "len": 3, "score": 2.0 + _spec(deg),
                "drug": drug_n, "disease": dis_n, "genes": [p1n, p2n], "pathway": None,
                "text": (f"{drug_n} --targets--> {p1n} --interacts--> "
                         f"{p2n} <--associated-- {dis_n}"),
            })

    # 3) shared pathway: drug target and a disease protein lie in one pathway.
    for p1 in targets:
        p1n = _name(data, PROT, p1)
        for pw in idx["prot2pw"].get(p1, set()):
            pw_sz = idx["pw_size"].get(pw, 1)
            if pw_sz > pw_size_cap:
                continue
            members = idx["pw2prot"].get(pw, set()) & dis_prots
            if not members:
                continue
            p2 = min(members, key=lambda q: idx["prot_dis_deg"].get(q, 1))
            pwn, p2n = _name(data, PATHWAY, pw), _name(data, PROT, p2)
            out.append({
                "type": "shared_pathway", "len": 4, "score": 1.0 + _spec(pw_sz),
                "drug": drug_n, "disease": dis_n, "genes": [p1n, p2n], "pathway": pwn,
                "text": (f"{drug_n} --targets--> {p1n} --in pathway--> "
                         f"{pwn} <--in pathway-- {p2n} <--associated-- {dis_n}"),
            })

    out.sort(key=lambda d: -d["score"])
    seen, uniq = set(), []
    for p in out:
        if p["text"] in seen:
            continue
        seen.add(p["text"])
        uniq.append(p)
    return uniq[:max_paths]


def classify_support(paths: List[Dict]) -> str:
    """Coarse, pre-LLM mechanism signal from path types (cheap triage)."""
    if any(p["type"] == "direct_target" for p in paths):
        return "direct-target mechanism"
    if any(p["type"] == "ppi" for p in paths):
        return "interaction-level mechanism"
    if any(p["type"] == "shared_pathway" for p in paths):
        return "pathway-level mechanism"
    return "no mechanistic path found"


def mechanism_score(paths: List[Dict]) -> float:
    """Single graph-only mechanism strength for a (drug, disease) pair.

    Used to test the falsifiable claim that true indications carry stronger
    mechanistic structure than random pairs. Higher = more/closer mechanism.
    """
    return max((p["score"] for p in paths), default=0.0)
