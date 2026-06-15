"""Graph-shuffle ablation: the falsification test.

If the GNN's advantage comes from the spatial ARRANGEMENT, then giving it a
graph with the same nodes/features but randomized edges must erase the
advantage and collapse it toward the composition-only baseline. We compare the
GNN on intact vs. shuffled vs. empty cell graphs.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .config import SplitConfig
from .train import train_any


def _shuffle_graph(g, rng):
    g2 = g.clone()
    n = g.x.size(0)
    m = g.edge_index.size(1) // 2
    src = rng.integers(0, n, size=m)
    dst = rng.integers(0, n, size=m)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    s = np.concatenate([src, dst]); d = np.concatenate([dst, src])
    g2.edge_index = torch.from_numpy(np.stack([s, d])).long()
    return g2


def _empty_graph(g):
    g2 = g.clone()
    g2.edge_index = torch.empty((2, 0), dtype=torch.long)
    return g2


def _perturb_cohort(ds, kind, seed):
    import copy as _copy
    rng = np.random.default_rng(seed)
    new = _copy.copy(ds)
    if kind == "intact":
        return ds
    if kind == "shuffled":
        new.graphs = [_shuffle_graph(g, rng) for g in ds.graphs]
    elif kind == "empty":
        new.graphs = [_empty_graph(g) for g in ds.graphs]
    else:
        raise ValueError(kind)
    return new


def run_graph_ablation(ds, make_split, model_cfg, train_cfg, seeds, verbose=True):
    conditions = ["intact", "shuffled", "empty"]
    out = {c: [] for c in conditions}
    comp_runs = []
    for seed in seeds:
        split = make_split(ds, SplitConfig(seed=seed))
        for c in conditions:
            pds = _perturb_cohort(ds, c, seed)
            out[c].append(train_any("sage", pds, split, model_cfg, train_cfg, seed=seed))
        comp_runs.append(train_any("xgboost", ds, split, model_cfg, train_cfg, seed=seed))
        if verbose:
            print("  seed=%d " % seed + " ".join(
                f"{c}={out[c][-1]['test']['macro_f1']:.3f}" for c in conditions)
                + f" xgb={comp_runs[-1]['test']['macro_f1']:.3f}")

    def agg(runs):
        return {k: {"mean": float(np.nanmean([r["test"][k] for r in runs])),
                    "std": float(np.nanstd([r["test"][k] for r in runs]))}
                for k in ("accuracy", "macro_f1", "auroc")}

    return {"conditions": {c: agg(out[c]) for c in conditions},
            "composition_reference": agg(comp_runs), "seeds": list(seeds)}


def print_summary(res):
    print("\n" + "=" * 60)
    print("GRAPH-SHUFFLE ABLATION (test macro-F1)")
    print("=" * 60)
    for c, a in res["conditions"].items():
        print(f"  GNN [{c:8s}] F1={a['macro_f1']['mean']:.3f}+/-{a['macro_f1']['std']:.2f}"
              f"  AUROC={a['auroc']['mean']:.3f}")
    r = res["composition_reference"]
    print(f"  XGB [compos.] F1={r['macro_f1']['mean']:.3f}+/-{r['macro_f1']['std']:.2f}"
          f"  AUROC={r['auroc']['mean']:.3f}")
