"""Graph-removal ablation: the falsification test.

If the GNN's advantage genuinely comes from the *spatial graph*, then giving it
a graph with the same number of edges but random (non-spatial) connectivity, or
no graph at all, must erase the advantage and collapse it toward the MLP. We run
the identical GNN on three graphs:

  * ``intact``   -- the true spatial kNN graph.
  * ``shuffled`` -- random edges, same count (spatial structure destroyed).
  * ``empty``    -- no edges (the GNN degenerates toward an MLP).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .splits import SpatialSplit
from .train import train_model


def shuffle_edges(edge_index, num_nodes, seed=0):
    rng = np.random.default_rng(seed)
    m = edge_index.size(1) // 2
    src = rng.integers(0, num_nodes, size=m)
    dst = rng.integers(0, num_nodes, size=m)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    s = np.concatenate([src, dst]); d = np.concatenate([dst, src])
    return torch.from_numpy(np.stack([s, d])).long()


def perturb(split: SpatialSplit, kind: str, seed: int) -> SpatialSplit:
    s = copy.copy(split)
    if kind == "intact":
        return s
    if kind == "shuffled":
        s.edge_index = shuffle_edges(split.edge_index, split.num_nodes, seed)
    elif kind == "empty":
        s.edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        raise ValueError(kind)
    return s


def run_graph_ablation(ds, make_split, split_cfg, model_cfg, train_cfg,
                       seeds, verbose=True) -> dict:
    conditions = ["intact", "shuffled", "empty"]
    out = {c: [] for c in conditions}
    mlp_runs = []
    for seed in seeds:
        from .config import SplitConfig
        split = make_split(ds, SplitConfig(**{**split_cfg.__dict__, "seed": seed}))
        for c in conditions:
            run = train_model("gnn", perturb(split, c, seed), model_cfg, train_cfg, seed=seed)
            out[c].append(run)
        mlp_runs.append(train_model("mlp", split, model_cfg, train_cfg, seed=seed))
        if verbose:
            print("  seed=%d " % seed + " ".join(
                f"{c}={out[c][-1]['test']['macro_f1']:.3f}" for c in conditions)
                + f" mlp={mlp_runs[-1]['test']['macro_f1']:.3f}")

    def agg(runs):
        return {k: {"mean": float(np.mean([r["test"][k] for r in runs])),
                    "std": float(np.std([r["test"][k] for r in runs]))}
                for k in ("accuracy", "macro_f1", "ari")}

    return {"conditions": {c: agg(out[c]) for c in conditions},
            "mlp_reference": agg(mlp_runs), "seeds": list(seeds)}


def print_summary(res):
    print("\n" + "=" * 60)
    print("GRAPH-REMOVAL ABLATION (test macro-F1)")
    print("=" * 60)
    for c, a in res["conditions"].items():
        print(f"  GNN [{c:8s}] F1={a['macro_f1']['mean']:.3f}+/-{a['macro_f1']['std']:.2f}"
              f"  ARI={a['ari']['mean']:.3f}")
    r = res["mlp_reference"]
    print(f"  MLP [blind   ] F1={r['macro_f1']['mean']:.3f}+/-{r['macro_f1']['std']:.2f}"
          f"  ARI={r['ari']['mean']:.3f}")
