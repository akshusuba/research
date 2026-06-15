#!/usr/bin/env python
"""Offline proof: does spatial arrangement beat composition for outcome?

Generates a synthetic patient cohort where the outcome label is set purely by
the spatial arrangement of immune cells (infiltration vs. exclusion) with cell
composition held identical across classes, then compares composition-only
baselines (LogReg/XGBoost/MLP) against a spatial GNN, plus a graph-shuffle
ablation.

Expected: composition baselines ~chance; spatial GNN >> baselines; GNN collapses
to baseline when the graph is shuffled/removed.

Usage:  python scripts/run_synthetic_demo.py [--fast]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_prognosis import ablation, experiment
from spatial_prognosis.config import (ModelConfig, SyntheticConfig, TrainConfig,
                                       RESULTS_DIR)
from spatial_prognosis.data.synthetic import generate_cohort
from spatial_prognosis.splits import make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        scfg = SyntheticConfig(n_patients=150, cells_per_patient=250)
        tcfg = TrainConfig(epochs=60, patience=15)
        seeds = [0, 1, 2]
    else:
        scfg = SyntheticConfig()
        tcfg = TrainConfig(epochs=150, patience=30)
        seeds = [0, 1, 2, 42, 123]
    mcfg = ModelConfig(encoder="sage")

    print("#" * 70)
    print("# spatial_prognosis -- synthetic cohort demonstration")
    print("#" * 70)

    comp = experiment.run_comparison(
        models=["logreg", "xgboost", "mlp", "sage"], seeds=seeds,
        synthetic_cfg=scfg, model_cfg=mcfg, train_cfg=tcfg)
    print("\nData stats:", json.dumps(comp["data_stats"]))
    experiment.print_summary(comp)
    p1 = experiment.save_results(comp, "synthetic_comparison.json")
    print(f"\nSaved -> {p1}")

    print("\nGraph-shuffle ablation...")
    ds = generate_cohort(scfg)
    abl = ablation.run_graph_ablation(ds, make_split, mcfg, tcfg, seeds)
    ablation.print_summary(abl)
    with open(os.path.join(RESULTS_DIR, "graph_ablation.json"), "w") as f:
        json.dump(abl, f, indent=2)
    print("Saved -> results/graph_ablation.json")


if __name__ == "__main__":
    main()
