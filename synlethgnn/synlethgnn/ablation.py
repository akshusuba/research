"""The headline falsification test: topology removal.

If the GNN's advantage truly comes from graph structure (and not from some
leakage through node features), then *destroying* the structure must erase the
advantage. We run the identical GNN on three graphs:

  * ``intact``   -- the real interaction graph.
  * ``rewired``  -- a random graph with the same node and edge count (community
                    structure destroyed, density preserved).
  * ``empty``    -- no edges at all (the GNN degenerates toward an MLP).

A genuine "GNN shines" result shows intact >> rewired ~ empty ~ MLP. This is
the single most convincing piece of evidence for the project's thesis.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch

from .config import (ModelConfig, SplitConfig, SyntheticConfig, TrainConfig,
                     RESULTS_DIR, SEEDS)
from .data.synthetic import generate_synthetic_sl
from .splits import make_split
from .train import train_model

METRIC_KEYS = ["auroc", "auprc", "hits@1", "mrr"]


def rewire_edges(edge_index: torch.Tensor, num_nodes: int,
                 seed: int = 0) -> torch.Tensor:
    """Erdos-Renyi rewiring: same edge count, random endpoints, symmetric."""
    rng = np.random.default_rng(seed)
    n_edges = edge_index.size(1) // 2  # undirected stored both ways
    src = rng.integers(0, num_nodes, size=n_edges)
    dst = rng.integers(0, num_nodes, size=n_edges)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    return torch.from_numpy(np.stack([s, d])).long()


def empty_edges() -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long)


def _apply_perturbation(split, kind: str, seed: int):
    """Return a copy of the split with perturbed message-passing graphs."""
    import copy
    s = copy.copy(split)
    if kind == "intact":
        return s
    if kind == "rewired":
        s.train_edge_index = rewire_edges(split.train_edge_index, split.num_nodes, seed)
        s.eval_edge_index = rewire_edges(split.eval_edge_index, split.num_nodes, seed + 1)
        return s
    if kind == "empty":
        s.train_edge_index = empty_edges()
        s.eval_edge_index = empty_edges()
        return s
    raise ValueError(kind)


def run_topology_ablation(mode: str = "transductive", seeds: List[int] = None,
                          synthetic_cfg: SyntheticConfig = None,
                          model_cfg: ModelConfig = None,
                          train_cfg: TrainConfig = None,
                          verbose: bool = True) -> dict:
    seeds = seeds or SEEDS
    synthetic_cfg = synthetic_cfg or SyntheticConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    conditions = ["intact", "rewired", "empty"]
    out: Dict[str, List[dict]] = {c: [] for c in conditions}
    mlp_runs: List[dict] = []

    for seed in seeds:
        scfg = SyntheticConfig(**{**synthetic_cfg.__dict__, "seed": seed})
        graph = generate_synthetic_sl(scfg)
        split = make_split(graph, SplitConfig(mode=mode, seed=seed),
                           neg_ratio=train_cfg.neg_ratio)
        # GNN under each perturbation
        for c in conditions:
            psplit = _apply_perturbation(split, c, seed)
            run = train_model("gnn", psplit, model_cfg, train_cfg, seed=seed)
            out[c].append(run)
        # MLP reference (structure-blind by construction)
        mlp_runs.append(train_model("mlp", split, model_cfg, train_cfg, seed=seed))
        if verbose:
            print(f"  seed={seed} "
                  + " ".join(f"{c}={out[c][-1]['test']['auprc']:.3f}"
                             for c in conditions)
                  + f" mlp={mlp_runs[-1]['test']['auprc']:.3f}")

    def agg(runs):
        return {k: {"mean": float(np.nanmean([r["test"][k] for r in runs])),
                    "std": float(np.nanstd([r["test"][k] for r in runs]))}
                for k in METRIC_KEYS}

    results = {
        "mode": mode, "seeds": seeds,
        "conditions": {c: agg(out[c]) for c in conditions},
        "mlp_reference": agg(mlp_runs),
        "config": {"synthetic": synthetic_cfg.__dict__,
                   "model": model_cfg.__dict__, "train": train_cfg.__dict__},
    }
    return results


def save_results(results: dict, name: str = "topology_ablation.json"):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def print_summary(results: dict):
    print("\n" + "=" * 64)
    print(f"TOPOLOGY-REMOVAL ABLATION ({results['mode']}, test AUPRC)")
    print("=" * 64)
    for c, a in results["conditions"].items():
        print(f"  GNN [{c:8s}]  AUPRC={a['auprc']['mean']:.3f}+/-{a['auprc']['std']:.2f}"
              f"   AUROC={a['auroc']['mean']:.3f}   MRR={a['mrr']['mean']:.3f}")
    r = results["mlp_reference"]
    print(f"  MLP [blind   ]  AUPRC={r['auprc']['mean']:.3f}+/-{r['auprc']['std']:.2f}"
          f"   AUROC={r['auroc']['mean']:.3f}   MRR={r['mrr']['mean']:.3f}")
