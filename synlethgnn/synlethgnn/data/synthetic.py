"""Controlled synthetic synthetic-lethality benchmark.

This module builds a gene-interaction graph in which synthetic lethality is,
*by construction*, a purely topological property: a pair (a, b) is lethal iff
the two genes belong to different redundant modules of the same essential
process. Node features are deliberately uninformative about SL.

Why this matters: it lets us make a falsifiable claim. If a model can predict
SL from this graph, it must be using topology. A feature-only MLP therefore
*cannot* beat chance, while a GNN that aggregates neighborhood structure can.
The same generator powers the topology-removal ablation (shuffle/strip edges
and the GNN must collapse to chance), which is the headline experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from ..config import SyntheticConfig


@dataclass
class SyntheticGraph:
    """Container bundling the PyG graph with ground-truth SL labels.

    Attributes
    ----------
    data : Data
        PyG graph holding ``x`` (node features) and ``edge_index`` (the gene
        interaction network used for message passing).
    sl_pos : (2, P) LongTensor
        Ground-truth synthetic-lethal gene pairs (undirected, stored once).
    sl_neg_pool : (2, N) LongTensor
        Hard negatives: non-lethal pairs drawn from the same processes
        (same-module pairs) plus cross-process pairs.
    gene_process : (num_nodes,) LongTensor
        Process id for each gene (metadata / diagnostics only).
    gene_module : (num_nodes,) LongTensor
        Global module id for each gene (metadata / diagnostics only).
    """

    data: Data
    sl_pos: torch.Tensor
    sl_neg_pool: torch.Tensor
    gene_process: torch.Tensor
    gene_module: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes


def _undirected(edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Turn a list of (u, v) into a symmetric edge_index tensor."""
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    arr = np.array(edges, dtype=np.int64).T
    src = np.concatenate([arr[0], arr[1]])
    dst = np.concatenate([arr[1], arr[0]])
    ei = np.unique(np.stack([src, dst]), axis=1)
    return torch.from_numpy(ei).long()


def generate_synthetic_sl(cfg: SyntheticConfig) -> SyntheticGraph:
    """Generate a topology-defined synthetic lethality graph."""
    rng = np.random.default_rng(cfg.seed)

    n_proc = cfg.n_processes
    mpp = cfg.modules_per_process
    gpm = cfg.genes_per_module
    n_modules = n_proc * mpp
    n_genes = n_modules * gpm

    # Assign each gene to a (process, module).
    gene_process = np.empty(n_genes, dtype=np.int64)
    gene_module = np.empty(n_genes, dtype=np.int64)
    module_genes: Dict[int, List[int]] = {m: [] for m in range(n_modules)}
    g = 0
    for p in range(n_proc):
        for m_local in range(mpp):
            m_global = p * mpp + m_local
            for _ in range(gpm):
                gene_process[g] = p
                gene_module[g] = m_global
                module_genes[m_global].append(g)
                g += 1

    # ----- Build the interaction graph (message-passing structure) -----
    edges: List[Tuple[int, int]] = []

    # 1) Dense intra-module cooperation edges.
    for m in range(n_modules):
        members = module_genes[m]
        for u, v in combinations(members, 2):
            if rng.random() < cfg.intra_module_p:
                edges.append((u, v))

    # 2) Sparse edges between redundant modules of the *same* process.
    for p in range(n_proc):
        proc_modules = [p * mpp + k for k in range(mpp)]
        for ma, mb in combinations(proc_modules, 2):
            for u in module_genes[ma]:
                for v in module_genes[mb]:
                    if rng.random() < cfg.inter_module_p:
                        edges.append((u, v))

    # 3) Background cross-process edges (noise).
    n_background = int(cfg.cross_process_p * n_genes * n_genes)
    for _ in range(n_background):
        u = int(rng.integers(n_genes))
        v = int(rng.integers(n_genes))
        if u != v and gene_process[u] != gene_process[v]:
            edges.append((u, v))

    edge_index = _undirected(edges)

    # ----- Node features: deliberately (almost) uninformative for SL -----
    x = rng.normal(0.0, 1.0, size=(n_genes, cfg.noise_features)).astype(np.float32)
    if cfg.feature_signal > 0.0:
        # Optionally leak a little process identity into features so the
        # synthetic study can be made less adversarial when desired.
        proc_onehot = np.zeros((n_genes, n_proc), dtype=np.float32)
        proc_onehot[np.arange(n_genes), gene_process] = 1.0
        x = np.concatenate([x, cfg.feature_signal * proc_onehot], axis=1)
    x = torch.from_numpy(x)

    # ----- Ground-truth SL pairs: different module, same process -----
    pos_pairs: List[Tuple[int, int]] = []
    for p in range(n_proc):
        proc_modules = [p * mpp + k for k in range(mpp)]
        for ma, mb in combinations(proc_modules, 2):
            for u in module_genes[ma]:
                for v in module_genes[mb]:
                    pos_pairs.append((u, v))

    # ----- Hard negatives: same-module pairs + cross-process pairs -----
    neg_pairs: List[Tuple[int, int]] = []
    # same-module (redundant, NOT lethal)
    for m in range(n_modules):
        for u, v in combinations(module_genes[m], 2):
            neg_pairs.append((u, v))
    # cross-process (unrelated, NOT lethal) -- sample to keep the pool sized
    # comparably to the same-module negatives without exploding combinatorially.
    target_cross = len(pos_pairs)
    tries = 0
    max_tries = target_cross * 10
    added = 0
    seen = set()
    while added < target_cross and tries < max_tries:
        u = int(rng.integers(n_genes))
        v = int(rng.integers(n_genes))
        tries += 1
        if u == v or gene_process[u] == gene_process[v]:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        neg_pairs.append(key)
        added += 1

    pos = torch.tensor(pos_pairs, dtype=torch.long).t().contiguous()
    neg = torch.tensor(neg_pairs, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, num_nodes=n_genes)

    return SyntheticGraph(
        data=data,
        sl_pos=pos,
        sl_neg_pool=neg,
        gene_process=torch.from_numpy(gene_process),
        gene_module=torch.from_numpy(gene_module),
    )


def summarize(graph: SyntheticGraph) -> dict:
    """Return a small dict of graph statistics for logging."""
    return {
        "num_genes": int(graph.num_nodes),
        "num_edges": int(graph.data.edge_index.size(1) // 2),
        "num_sl_positive": int(graph.sl_pos.size(1)),
        "num_sl_negative_pool": int(graph.sl_neg_pool.size(1)),
        "avg_degree": float(graph.data.edge_index.size(1) / max(graph.num_nodes, 1)),
    }
