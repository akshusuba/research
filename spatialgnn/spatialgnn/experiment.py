"""Multi-seed model comparison for spatial domain classification."""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from .config import (ModelConfig, SplitConfig, SyntheticConfig, TrainConfig,
                     RESULTS_DIR, SEEDS)
from .data.synthetic import generate_synthetic_spatial, summarize
from .splits import make_split
from .train import train_any

DEFAULT_MODELS = ["xgboost", "mlp", "sage"]
METRIC_KEYS = ["accuracy", "macro_f1", "ari"]


def aggregate(runs: List[dict], section="test") -> dict:
    agg = {}
    for k in METRIC_KEYS:
        vals = [r[section][k] for r in runs]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "values": [float(v) for v in vals]}
    return agg


def run_comparison(models=None, modes=("cross_slice", "within_slice"),
                   seeds=None, synthetic_cfg=None, model_cfg=None,
                   train_cfg=None, split_cfg=None, verbose=True) -> dict:
    models = models or DEFAULT_MODELS
    seeds = seeds or SEEDS
    synthetic_cfg = synthetic_cfg or SyntheticConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    base_split = split_cfg or SplitConfig()

    ds = generate_synthetic_spatial(synthetic_cfg)
    results = {"graph_stats": summarize(ds),
               "config": {"synthetic": synthetic_cfg.__dict__,
                          "model": model_cfg.__dict__, "train": train_cfg.__dict__,
                          "models": models, "modes": list(modes), "seeds": seeds},
               "by_mode": {}}

    for mode in modes:
        if verbose:
            print(f"\n=== Split mode: {mode} ===")
        per_model = {m: [] for m in models}
        for seed in seeds:
            split = make_split(ds, SplitConfig(**{**base_split.__dict__,
                                                  "mode": mode, "seed": seed}))
            for m in models:
                run = train_any(m, split, model_cfg, train_cfg, seed=seed)
                per_model[m].append(run)
                if verbose:
                    t = run["test"]
                    print(f"  seed={seed} {m:8s} acc={t['accuracy']:.3f} "
                          f"F1={t['macro_f1']:.3f} ARI={t['ari']:.3f}")
        results["by_mode"][mode] = {m: {"agg": aggregate(per_model[m]),
                                        "runs": per_model[m]} for m in models}
    return results


def save_results(results, name="synthetic_comparison.json"):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def print_summary(results):
    print("\n" + "=" * 64)
    print("SUMMARY (test, mean +/- std over seeds)")
    print("=" * 64)
    for mode, mr in results["by_mode"].items():
        print(f"\n[{mode.upper()}]")
        print(f"{'model':9s} " + " ".join(f"{k:>14s}" for k in ["accuracy", "macro_f1", "ari"]))
        for m, d in mr.items():
            a = d["agg"]
            print(f"{m:9s} " + " ".join(
                f"{a[k]['mean']:.3f}+/-{a[k]['std']:.2f}" for k in ["accuracy", "macro_f1", "ari"]))
