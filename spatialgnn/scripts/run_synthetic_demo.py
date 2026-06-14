#!/usr/bin/env python
"""Offline demonstration that the spatial GNN earns its keep.

Runs on the controlled synthetic tissue benchmark (no downloads):
  1. Model comparison (XGBoost vs MLP vs spatial GNN) under cross-slice and
     within-slice splits.
  2. The graph-removal ablation (intact vs shuffled vs empty spatial graph).

Expected pattern:
  * GNN >> MLP ~ XGBoost  (the domain label is a neighbourhood property)
  * GNN(intact) >> GNN(shuffled) ~ GNN(empty) ~ MLP  (the win is spatial)

Usage:  python scripts/run_synthetic_demo.py [--fast]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialgnn import ablation, experiment
from spatialgnn.config import (ModelConfig, SplitConfig, SyntheticConfig,
                               TrainConfig)
from spatialgnn.data.synthetic import generate_synthetic_spatial
from spatialgnn.splits import make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    if args.fast:
        scfg = SyntheticConfig(n_slices=5, grid_size=22)
        tcfg = TrainConfig(epochs=120, patience=25)
        seeds = [0, 1, 2]
    else:
        scfg = SyntheticConfig()
        tcfg = TrainConfig(epochs=300, patience=40)
        seeds = [0, 1, 2, 42, 123]
    mcfg = ModelConfig(encoder="sage")
    split_cfg = SplitConfig()

    print("#" * 70)
    print("# SpatialGNN -- synthetic tissue demonstration")
    print("#" * 70)

    comp = experiment.run_comparison(
        models=["xgboost", "mlp", "sage"],
        modes=("cross_slice", "within_slice"), seeds=seeds,
        synthetic_cfg=scfg, model_cfg=mcfg, train_cfg=tcfg, split_cfg=split_cfg)
    experiment.print_summary(comp)
    p1 = experiment.save_results(comp, "synthetic_comparison.json")
    print(f"\nSaved comparison -> {p1}")

    print("\nGraph-removal ablation (cross_slice)...")
    ds = generate_synthetic_spatial(scfg)
    abl = ablation.run_graph_ablation(
        ds, make_split, SplitConfig(mode="cross_slice"), mcfg, tcfg, seeds)
    ablation.print_summary(abl)
    import json
    from spatialgnn.config import RESULTS_DIR
    with open(os.path.join(RESULTS_DIR, "graph_ablation.json"), "w") as f:
        json.dump(abl, f, indent=2)
    print("Saved ablation -> results/graph_ablation.json")


if __name__ == "__main__":
    main()
