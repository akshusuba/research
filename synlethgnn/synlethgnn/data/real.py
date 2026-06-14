"""Real synthetic-lethality data from SynLethDB via the KG4SL release.

KG4SL (Zheng et al., Bioinformatics 2021) packages SynLethDB SL pairs together
with a biomedical knowledge graph. We download three files and build a clean,
leakage-free gene graph:

  * ``SL_GsG``   triples  -> synthetic-lethal positives (the labels).
  * ``NONSL_GnsG`` triples -> experimentally non-lethal hard negatives.
  * gene-gene interaction relations (``INTERACTS_GiG``, ``REGULATES_GrG``,
    ``COVARIES_GcG``) -> the message-passing graph.

Crucially, the SL/NONSL relations are *excluded* from the message-passing graph,
so the model never sees the answer in its edges. Node features default to
degree + noise (uninformative about SL on their own), keeping the comparison
honest: anything the GNN gains over the MLP must come from topology.

Data source (branch ``main``):
  https://github.com/JieZheng-ShanghaiTech/KG4SL/tree/main/data
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

RAW_BASE = "https://raw.githubusercontent.com/JieZheng-ShanghaiTech/KG4SL/main/data"
FILES = {
    "kg_triplet.csv": f"{RAW_BASE}/kg_triplet.csv",
    "dbid2name.csv": f"{RAW_BASE}/dbid2name.csv",
    "relation2id.csv": f"{RAW_BASE}/relation2id.csv",
}

SL_RELATION = "SL_GsG"
NONSL_RELATION = "NONSL_GnsG"
# Ordered so each interaction relation gets a stable integer type id for R-GCN.
INTERACTION_RELATIONS = ["INTERACTS_GiG", "REGULATES_GrG", "COVARIES_GcG"]
# Gene -> functional-annotation relations used to build real biological
# features (GO biological process / molecular function / cellular component /
# pathway membership). The gene is the source (``a``) in these triples.
ANNOTATION_RELATIONS = ["PARTICIPATES_GpBP", "PARTICIPATES_GpMF",
                        "PARTICIPATES_GpCC", "PARTICIPATES_GpPW"]


@dataclass
class SLGraph:
    """Duck-typed to match SyntheticGraph for splits/training/eval.

    ``data.edge_type`` (when present) holds an integer relation id per edge so a
    relation-typed encoder (R-GCN) can use it; ``num_relations`` records how
    many distinct relation types exist (counting both directions).
    """

    data: Data
    sl_pos: torch.Tensor
    sl_neg_pool: torch.Tensor
    id2name: Dict[int, str]
    num_relations: int = 1
    feature_mode: str = "functional"

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes


def download_kg4sl(data_dir: str) -> str:
    """Download the KG4SL files (idempotent). Returns the destination dir."""
    import requests

    dest = os.path.join(data_dir, "kg4sl")
    os.makedirs(dest, exist_ok=True)
    for fname, url in FILES.items():
        path = os.path.join(dest, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"  [cached] {fname}")
            continue
        print(f"  downloading {fname} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"    -> {os.path.getsize(path) / 1e6:.1f} MB")
    return dest


def _load_id2name(dest: str) -> Dict[int, str]:
    id2name: Dict[int, str] = {}
    path = os.path.join(dest, "dbid2name.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                id2name[int(row["_id"])] = row["name"]
            except (KeyError, ValueError):
                continue
    return id2name


def _parse_triplets(dest: str):
    """Stream kg_triplet.csv and collect SL, NONSL, interaction (by relation),
    and gene->annotation edges."""
    inter_set = set(INTERACTION_RELATIONS)
    annot_set = set(ANNOTATION_RELATIONS)
    sl_pairs: List[Tuple[int, int]] = []
    nonsl_pairs: List[Tuple[int, int]] = []
    inter_by_rel: Dict[str, List[Tuple[int, int]]] = {r: [] for r in INTERACTION_RELATIONS}
    annot_pairs: List[Tuple[int, int]] = []   # (gene, annotation-entity)
    path = os.path.join(dest, "kg_triplet.csv")
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 3:
                continue
            a, rel, b = row
            try:
                a, b = int(a), int(b)
            except ValueError:
                continue
            if rel == SL_RELATION:
                sl_pairs.append((a, b))
            elif rel == NONSL_RELATION:
                nonsl_pairs.append((a, b))
            elif rel in inter_set:
                inter_by_rel[rel].append((a, b))
            elif rel in annot_set:
                annot_pairs.append((a, b))
    return sl_pairs, nonsl_pairs, inter_by_rel, annot_pairs


def _degree_bins(edge_index, n_nodes, n_bins=20):
    """Assign each node to a degree quantile bin; return (bin_of_node, bins)."""
    deg = np.zeros(n_nodes, dtype=np.int64)
    if edge_index.size(1) > 0:
        ei = edge_index[0].numpy()
        np.add.at(deg, ei, 1)
    # Quantile edges over the (log) degree so bins are balanced.
    ranks = np.argsort(np.argsort(deg))
    bin_of = (ranks * n_bins // max(n_nodes, 1)).clip(0, n_bins - 1)
    bins = {b: np.where(bin_of == b)[0] for b in range(n_bins)}
    return bin_of, bins


def _sample_degree_matched_negatives(sl_pos, edge_index, n_nodes, n_samples,
                                     exclude, rng, n_bins=20):
    """Sample non-SL pairs whose endpoint *degrees* match the positives.

    This removes the "popularity shortcut": with uniform-random negatives, SL
    genes (well-studied hubs) have systematically higher degree than random
    genes, so a model can cheat using degree alone. Matching the degree
    distribution of negatives to positives forces a model to rely on *which*
    genes interact (topology), not merely *how connected* they are.
    """
    bin_of, bins = _degree_bins(edge_index, n_nodes, n_bins)
    pos_u = sl_pos[0].numpy()
    pos_v = sl_pos[1].numpy()
    neg = []
    seen = set()
    n_pos = len(pos_u)
    for i in range(n_samples):
        # Pick a template positive and resample endpoints from the same bins.
        t = i % n_pos
        bu, bv = bin_of[pos_u[t]], bin_of[pos_v[t]]
        for _ in range(10):
            cu = bins[bu]; cv = bins[bv]
            if len(cu) == 0 or len(cv) == 0:
                break
            u = int(cu[rng.integers(len(cu))])
            v = int(cv[rng.integers(len(cv))])
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in seen or key in exclude:
                continue
            seen.add(key)
            neg.append(key)
            break
    if not neg:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(neg, dtype=torch.long).t().contiguous()


def _sample_random_negatives(sl_pos, n_nodes, n_samples, exclude, rng):
    """Sample random gene pairs that are not known SL pairs (open-world
    negatives, as used by SynLethDB/KG4SL). ``exclude`` is a set of frozensets
    of known positive pairs."""
    neg = []
    seen = set()
    tries = 0
    max_tries = n_samples * 20
    while len(neg) < n_samples and tries < max_tries:
        u = int(rng.integers(n_nodes))
        v = int(rng.integers(n_nodes))
        tries += 1
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen or key in exclude:
            continue
        seen.add(key)
        neg.append(key)
    if not neg:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(neg, dtype=torch.long).t().contiguous()


def _build_functional_features(annot_pairs, remap, n_genes, svd_dim, seed):
    """Build a dense gene feature matrix from GO/pathway membership.

    Each gene's raw fingerprint is a sparse multi-hot over the functional-
    annotation entities it participates in (GO BP/MF/CC, pathways). We TF-IDF
    weight it (down-weighting ubiquitous terms) and reduce to ``svd_dim`` dense
    dimensions with truncated SVD. These are genuine biological features, so the
    MLP/XGBoost baselines have real signal to use -- making the test of whether
    the *graph* adds value a fair and hard one.
    """
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer

    annot_remap: Dict[int, int] = {}
    rows, cols = [], []
    for g, ann in annot_pairs:
        if g not in remap:
            continue
        gi = remap[g]
        if ann not in annot_remap:
            annot_remap[ann] = len(annot_remap)
        rows.append(gi)
        cols.append(annot_remap[ann])
    n_ann = max(len(annot_remap), 1)
    data = np.ones(len(rows), dtype=np.float32)
    M = csr_matrix((data, (rows, cols)), shape=(n_genes, n_ann))
    coverage = float((np.asarray((M.sum(axis=1) > 0)).ravel()).mean()) if len(rows) else 0.0
    M = TfidfTransformer().fit_transform(M)
    dim = int(min(svd_dim, max(2, n_ann - 1)))
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    feats = svd.fit_transform(M).astype(np.float32)
    return torch.from_numpy(feats), coverage


def build_real_graph(data_dir: str, feature_mode: str = "functional",
                     svd_dim: int = 128, noise_dim: int = 16,
                     include_degree: bool = False, seed: int = 0,
                     neg_strategy: str = "degree_matched",
                     random_neg_ratio: float = 1.0,
                     relation_typed: bool = True) -> SLGraph:
    """Build the gene SL graph from cached KG4SL files.

    Parameters
    ----------
    feature_mode : {"functional", "noise"}
        "functional" (default) derives real GO/pathway features (see
        :func:`_build_functional_features`), giving the feature baselines a fair
        chance and making "does the graph add value beyond features?" the real
        question. "noise" uses uninformative Gaussian features (adversarial to
        feature models; isolates topology).
    relation_typed : bool
        If True, keep a distinct relation id per interaction type (plus reverse)
        in ``data.edge_type`` so an R-GCN can use it.
    neg_strategy : {"degree_matched", "random", "nonsl", "both"}
        Negative pool construction. "degree_matched" (default) removes the
        degree/popularity shortcut so the comparison tests topology.
    """
    dest = os.path.join(data_dir, "kg4sl")
    if not os.path.exists(os.path.join(dest, "kg_triplet.csv")):
        raise FileNotFoundError(
            "KG4SL files not found. Run download_kg4sl(data_dir) first."
        )

    id2name = _load_id2name(dest)
    sl_pairs, nonsl_pairs, inter_by_rel, annot_pairs = _parse_triplets(dest)
    all_inter = [e for r in INTERACTION_RELATIONS for e in inter_by_rel[r]]

    # Gene node set = every gene that appears in any SL/NONSL/interaction edge.
    gene_ids: Set[int] = set()
    for u, v in sl_pairs + nonsl_pairs + all_inter:
        gene_ids.add(u); gene_ids.add(v)
    sorted_ids = sorted(gene_ids)
    remap = {g: i for i, g in enumerate(sorted_ids)}
    n = len(sorted_ids)

    def remap_pairs(pairs):
        arr = [(remap[u], remap[v]) for u, v in pairs
               if u in remap and v in remap and u != v]
        if not arr:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(arr, dtype=torch.long).t().contiguous()

    sl_pos = remap_pairs(sl_pairs)
    explicit_neg = remap_pairs(nonsl_pairs)

    # Message-passing edges with relation ids (symmetrized; reverse edges get
    # their own relation id, the standard R-GCN convention).
    src_list, dst_list, type_list = [], [], []
    for rid, rel in enumerate(INTERACTION_RELATIONS):
        e = remap_pairs(inter_by_rel[rel])
        if e.size(1) == 0:
            continue
        # forward (type rid) and reverse (type rid + R)
        src_list += [e[0], e[1]]
        dst_list += [e[1], e[0]]
        nrel = len(INTERACTION_RELATIONS)
        type_list += [torch.full((e.size(1),), rid, dtype=torch.long),
                      torch.full((e.size(1),), rid + nrel, dtype=torch.long)]
    if src_list:
        edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)])
        edge_type = torch.cat(type_list)
        # dedup identical (src,dst,type) triples
        keyed = edge_index[0] * (n + 1) * 100 + edge_index[1] * 100 + edge_type
        _, idx = np.unique(keyed.numpy(), return_index=True)
        idx = torch.from_numpy(np.sort(idx))
        edge_index = edge_index[:, idx]
        edge_type = edge_type[idx]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    num_relations = 2 * len(INTERACTION_RELATIONS)

    # Negative pool (degree-matched by default to remove the popularity shortcut).
    rng = np.random.default_rng(seed)
    exclude = {(min(int(a), int(b)), max(int(a), int(b)))
               for a, b in zip(sl_pos[0].tolist(), sl_pos[1].tolist())}
    exclude |= {(min(int(a), int(b)), max(int(a), int(b)))
                for a, b in zip(explicit_neg[0].tolist(), explicit_neg[1].tolist())}
    if neg_strategy == "nonsl":
        sl_neg_pool = explicit_neg
    else:
        n_neg = int(random_neg_ratio * sl_pos.size(1))
        if neg_strategy == "degree_matched":
            sampled = _sample_degree_matched_negatives(
                sl_pos, edge_index, n, n_neg, exclude, rng)
        else:
            sampled = _sample_random_negatives(sl_pos, n, n_neg, exclude, rng)
        if neg_strategy == "both" and explicit_neg.size(1) > 0:
            sl_neg_pool = torch.cat([explicit_neg, sampled], dim=1)
        else:
            sl_neg_pool = sampled

    # Node features.
    if feature_mode == "functional":
        x, coverage = _build_functional_features(annot_pairs, remap, n, svd_dim, seed)
    else:
        feat_rng = np.random.default_rng(seed + 1)
        x = torch.from_numpy(feat_rng.normal(0, 1, size=(n, noise_dim)).astype(np.float32))
        coverage = 0.0

    if include_degree:
        deg = torch.zeros(n)
        if edge_index.size(1) > 0:
            deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        x = torch.cat([torch.log1p(deg).unsqueeze(1), x], dim=1)

    data = Data(x=x, edge_index=edge_index, num_nodes=n)
    if relation_typed:
        data.edge_type = edge_type
    data.feature_coverage = coverage
    id2name_remapped = {remap[g]: id2name.get(g, str(g)) for g in sorted_ids}
    return SLGraph(data=data, sl_pos=sl_pos, sl_neg_pool=sl_neg_pool,
                   id2name=id2name_remapped,
                   num_relations=num_relations if relation_typed else 1,
                   feature_mode=feature_mode)


def summarize(graph: SLGraph) -> dict:
    return {
        "num_genes": int(graph.num_nodes),
        "num_interaction_edges": int(graph.data.edge_index.size(1) // 2),
        "num_sl_positive": int(graph.sl_pos.size(1)),
        "num_negatives": int(graph.sl_neg_pool.size(1)),
        "feature_dim": int(graph.data.x.size(1)),
        "feature_mode": graph.feature_mode,
        "feature_coverage": round(float(getattr(graph.data, "feature_coverage", 0.0)), 3),
        "num_relations": int(graph.num_relations),
        "avg_degree": round(float(graph.data.edge_index.size(1) / max(graph.num_nodes, 1)), 2),
    }
