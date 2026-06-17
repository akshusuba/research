"""Supervision for the mechanism-recovery experiment.

For each (drug, disease) pair that DrugMechDB covers, we know the curated bridge
gene(s) (the protein the drug acts on to treat the disease). This module maps those
to PrimeKG ``gene_protein`` node indices and provides degree-matched decoy genes,
so a GNN can be trained to rank the true bridge gene above decoys.

LEAKAGE RULE: callers must build training supervision only from TRAINING-split
pairs (never from held-out/test diseases or drugs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE
from oncorepurpose.interpret.uniprot_map import uniprot_to_symbol

GENE_TYPE = "gene_protein"
_DMDB_URLS = (
    "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.yaml",
    "https://raw.githubusercontent.com/SuLab/DrugMechDB/master/indication_paths.yaml",
)


def build_drugmechdb_drug_symbols(cache: Optional[Path] = None) -> Dict[str, set]:
    """drug name (lower) -> set of curated HGNC MOA gene symbols (UniProt-mapped)."""
    import requests
    import yaml
    raw = None
    for u in _DMDB_URLS:
        try:
            r = requests.get(u, timeout=60)
            if r.ok and len(r.text) > 1000:
                raw = r.text
                break
        except Exception:
            continue
    if raw is None:
        return {}
    entries = yaml.safe_load(raw)
    accs, drug_accs = set(), {}
    for e in entries:
        drug = str(e.get("graph", {}).get("drug", "")).strip().lower()
        if not drug:
            continue
        for n in e.get("nodes", []):
            nid = str(n.get("id", ""))
            if nid.startswith("UniProt:"):
                a = nid.split(":", 1)[1]
                accs.add(a)
                drug_accs.setdefault(drug, set()).add(a)
    mp = uniprot_to_symbol(sorted(accs))
    out: Dict[str, set] = {}
    for drug, a_set in drug_accs.items():
        syms = {mp[a].upper() for a in a_set if mp.get(a)}
        if syms:
            out[drug] = syms
    return out


def symbol_to_gene_index(data: HeteroData) -> Dict[str, int]:
    names = list(data[GENE_TYPE].node_names)
    out: Dict[str, int] = {}
    for i, nm in enumerate(names):
        s = str(nm).strip().upper()
        if s and s not in out:
            out[s] = i
    return out


@dataclass
class MechExamples:
    drug: torch.Tensor      # [M] drug node idx
    dis: torch.Tensor       # [M] disease node idx
    gene: torch.Tensor      # [M] one positive bridge gene idx (expanded per gene)
    pairs: list             # list of (drug_idx, dis_idx, [gene_idx,...]) unique pairs


def build_mech_examples(
    data: HeteroData, pair_index: torch.Tensor,
    dmdb: Dict[str, set], sym2gidx: Dict[str, int],
) -> MechExamples:
    """From positive (drug, disease) columns, build (drug, dis, bridge-genes) examples
    for the pairs DrugMechDB covers and whose genes exist in PrimeKG."""
    drug_names = [str(x).strip().lower() for x in data[DRUG_TYPE].node_names]
    drugs, diss, genes, pairs = [], [], [], []
    seen = set()
    for c in range(pair_index.size(1)):
        di = int(pair_index[0, c]); ci = int(pair_index[1, c])
        if (di, ci) in seen:
            continue
        seen.add((di, ci))
        syms = dmdb.get(drug_names[di]) if di < len(drug_names) else None
        if not syms:
            continue
        gidx = sorted({sym2gidx[s] for s in syms if s in sym2gidx})
        if not gidx:
            continue
        pairs.append((di, ci, gidx))
        for g in gidx:
            drugs.append(di); diss.append(ci); genes.append(g)
    return MechExamples(
        torch.tensor(drugs, dtype=torch.long),
        torch.tensor(diss, dtype=torch.long),
        torch.tensor(genes, dtype=torch.long),
        pairs,
    )


class DegreeMatchedDecoys:
    """Sample decoy genes from the same drug-degree bucket as a positive gene,
    so the model cannot rank the true bridge gene by hub degree alone."""

    def __init__(self, prot_drug_deg: Dict[int, int], num_genes: int, n_buckets: int = 10, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        deg = np.array([prot_drug_deg.get(i, 0) for i in range(num_genes)], dtype=float)
        # Bucket by rank so buckets are balanced even with a skewed degree dist.
        order = np.argsort(deg, kind="stable")
        bucket = np.empty(num_genes, dtype=int)
        edges = np.linspace(0, num_genes, n_buckets + 1).astype(int)
        for b in range(n_buckets):
            bucket[order[edges[b]:edges[b + 1]]] = b
        self.bucket = bucket
        self.by_bucket = [np.where(bucket == b)[0] for b in range(n_buckets)]

    def sample(self, pos_gene: int, exclude: set, k: int) -> List[int]:
        b = self.bucket[pos_gene]
        pool = self.by_bucket[b]
        out, tries = [], 0
        while len(out) < k and tries < k * 50:
            cand = int(pool[self.rng.integers(0, len(pool))])
            if cand not in exclude and cand not in out:
                out.append(cand)
            tries += 1
        # Backfill from anywhere if the bucket is too small.
        while len(out) < k:
            cand = int(self.rng.integers(0, len(self.bucket)))
            if cand not in exclude and cand not in out:
                out.append(cand)
        return out
