#!/usr/bin/env python
"""Offline demonstration that the GNN earns its keep on synthetic lethality.

Runs entirely on the controlled synthetic benchmark (no downloads), so it
reproduces the core thesis in a couple of minutes on CPU:

  1. Model comparison (MLP vs node2vec vs GNN) under transductive AND
     inductive (cold-gene) splits.
  2. The topology-removal ablation (intact vs rewired vs empty graph).

Expected pattern:
  * transductive: GNN ~ node2vec >> MLP  (structure matters)
  * inductive   : GNN >> node2vec ~ MLP  (only the GNN generalizes structurally)
  * ablation    : GNN(intact) >> GNN(rewired) ~ GNN(empty) ~ MLP  (the win is topological)

Usage:
    python scripts/run_synthetic_demo.py [--fast]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synlethgnn import ablation, experiment
from synlethgnn.config import ModelConfig, SyntheticConfig, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="smaller graph + fewer seeds/epochs for a quick smoke test")
    args = parser.parse_args()

    if args.fast:
        synthetic_cfg = SyntheticConfig(n_processes=20, modules_per_process=3,
                                        genes_per_module=6)
        train_cfg = TrainConfig(epochs=200, patience=40)
        seeds = [0, 1, 2]
    else:
        synthetic_cfg = SyntheticConfig()
        train_cfg = TrainConfig(epochs=300, patience=40)
        seeds = [0, 1, 2, 42, 123]

    model_cfg = ModelConfig(encoder="sage")

    print("#" * 70)
    print("# SynLethGNN -- synthetic-data demonstration")
    print("#" * 70)

    comp = experiment.run_comparison(
        models=["mlp", "node2vec", "gnn"],
        modes=("transductive", "inductive"),
        seeds=seeds, synthetic_cfg=synthetic_cfg,
        model_cfg=model_cfg, train_cfg=train_cfg, verbose=True,
    )
    experiment.print_summary(comp)
    p1 = experiment.save_results(comp, "synthetic_comparison.json")
    print(f"\nSaved comparison -> {p1}")

    print("\nRunning topology-removal ablation (transductive)...")
    abl = ablation.run_topology_ablation(
        mode="transductive", seeds=seeds, synthetic_cfg=synthetic_cfg,
        model_cfg=model_cfg, train_cfg=train_cfg, verbose=True,
    )
    ablation.print_summary(abl)
    p2 = ablation.save_results(abl, "topology_ablation.json")
    print(f"\nSaved ablation -> {p2}")


if __name__ == "__main__":
    main()
