"""Orchestration: multi-seed model comparison across split regimes.

Produces the core result table -- every model x {transductive, inductive} x
seeds -- with mean +/- std, mirroring the rigor bar used in the celiac project
(multi-seed, honest baselines, ranking + threshold metrics).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .config import (ExperimentConfig, ModelConfig, SplitConfig, SyntheticConfig,
                     TrainConfig, RESULTS_DIR, SEEDS)
from .data.synthetic import generate_synthetic_sl, summarize
from .splits import make_split
from .train import train_model

DEFAULT_MODELS = ["mlp", "node2vec", "gnn"]
METRIC_KEYS = ["auroc", "auprc", "hits@1", "hits@3", "hits@10", "mrr"]


def aggregate(runs: List[dict], section: str = "test") -> Dict[str, dict]:
    """Mean/std across seeds for each metric."""
    agg = {}
    for key in METRIC_KEYS:
        vals = [r[section][key] for r in runs if not np.isnan(r[section][key])]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "values": [float(v) for v in vals]}
        else:
            agg[key] = {"mean": float("nan"), "std": float("nan"), "values": []}
    return agg


def run_comparison(models=None, modes=("transductive", "inductive"),
                   seeds=None, synthetic_cfg: SyntheticConfig = None,
                   model_cfg: ModelConfig = None, train_cfg: TrainConfig = None,
                   verbose: bool = True) -> dict:
    models = models or DEFAULT_MODELS
    seeds = seeds or SEEDS
    synthetic_cfg = synthetic_cfg or SyntheticConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    results = {"config": {
        "synthetic": synthetic_cfg.__dict__,
        "model": model_cfg.__dict__,
        "train": train_cfg.__dict__,
        "models": models, "modes": list(modes), "seeds": seeds,
    }, "graph_stats": None, "by_mode": {}}

    for mode in modes:
        if verbose:
            print(f"\n=== Split mode: {mode} ===")
        mode_runs: Dict[str, List[dict]] = {m: [] for m in models}
        for seed in seeds:
            # Regenerate graph per seed for an honest multi-seed estimate.
            scfg = SyntheticConfig(**{**synthetic_cfg.__dict__, "seed": seed})
            graph = generate_synthetic_sl(scfg)
            if results["graph_stats"] is None:
                results["graph_stats"] = summarize(graph)
            split = make_split(graph, SplitConfig(mode=mode, seed=seed),
                               neg_ratio=train_cfg.neg_ratio)
            for m in models:
                run = train_model(m, split, model_cfg, train_cfg, seed=seed)
                mode_runs[m].append(run)
                if verbose:
                    t = run["test"]
                    print(f"  seed={seed} {m:9s} "
                          f"AUROC={t['auroc']:.3f} AUPRC={t['auprc']:.3f} "
                          f"Hits@1={t['hits@1']:.3f} MRR={t['mrr']:.3f}")
        results["by_mode"][mode] = {
            m: {"agg": aggregate(mode_runs[m]), "runs": mode_runs[m]}
            for m in models
        }

    return results


def save_results(results: dict, name: str = "synthetic_comparison.json"):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def print_summary(results: dict):
    print("\n" + "=" * 64)
    print("SUMMARY (test set, mean +/- std over seeds)")
    print("=" * 64)
    for mode, mode_res in results["by_mode"].items():
        print(f"\n[{mode.upper()}]")
        header = f"{'model':10s} " + " ".join(f"{k:>12s}" for k in
                                              ["auroc", "auprc", "hits@1", "mrr"])
        print(header)
        for m, mr in mode_res.items():
            agg = mr["agg"]
            row = f"{m:10s} " + " ".join(
                f"{agg[k]['mean']:.3f}+/-{agg[k]['std']:.2f}"
                for k in ["auroc", "auprc", "hits@1", "mrr"])
            print(row)
