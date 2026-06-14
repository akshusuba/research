#!/usr/bin/env python
"""Run the SynLethGNN comparison on real SynLethDB (KG4SL) data.

Downloads the KG4SL files (~65 MB, cached), builds a leakage-free gene
interaction graph with real GO/pathway features and relation-typed edges, and
compares strong baselines (XGBoost, MLP, node2vec) against graph models
(GraphSAGE, relation-typed R-GCN) under transductive and inductive (cold-gene)
splits.

The decisive question: does message passing add value *beyond* a powerful
tabular learner (XGBoost) that already sees real biological features?

Usage:
    python scripts/run_real.py                      # full comparison
    python scripts/run_real.py --no-download
    python scripts/run_real.py --feature-mode noise # adversarial-to-features
    python scripts/run_real.py --seeds 0 1 --models xgboost mlp sage rgcn
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synlethgnn.config import (DATA_DIR, RESULTS_DIR, ModelConfig, SplitConfig,
                               TrainConfig)
from synlethgnn.data import real as realdata
from synlethgnn.experiment import aggregate
from synlethgnn.splits import make_split
from synlethgnn.train import train_any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models", nargs="+",
                        default=["xgboost", "mlp", "node2vec", "sage", "rgcn"])
    parser.add_argument("--modes", nargs="+",
                        default=["transductive", "inductive"])
    parser.add_argument("--feature-mode", default="functional",
                        choices=["functional", "noise"])
    parser.add_argument("--neg-strategy", default="degree_matched")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--restarts", type=int, default=2)
    args = parser.parse_args()

    if not args.no_download:
        print("Downloading KG4SL data...")
        realdata.download_kg4sl(DATA_DIR)

    print("Building real gene SL graph...")
    graph = realdata.build_real_graph(
        DATA_DIR, feature_mode=args.feature_mode, neg_strategy=args.neg_strategy,
        relation_typed=True)
    stats = realdata.summarize(graph)
    print("Graph stats:", json.dumps(stats, indent=2))

    model_cfg = ModelConfig()
    train_cfg = TrainConfig(epochs=args.epochs, patience=40,
                            num_restarts=args.restarts)

    results = {"graph_stats": stats, "config": {
        "feature_mode": args.feature_mode, "neg_strategy": args.neg_strategy,
        "epochs": args.epochs, "restarts": args.restarts, "seeds": args.seeds,
        "models": args.models}, "by_mode": {}}

    for mode in args.modes:
        print(f"\n=== {mode} ===")
        per_model = {m: [] for m in args.models}
        for seed in args.seeds:
            split = make_split(graph, SplitConfig(mode=mode, seed=seed),
                               neg_ratio=train_cfg.neg_ratio)
            for m in args.models:
                run = train_any(m, split, model_cfg, train_cfg, seed=seed)
                per_model[m].append(run)
                t = run["test"]
                print(f"  seed={seed} {m:9s} AUROC={t['auroc']:.3f} "
                      f"AUPRC={t['auprc']:.3f} Hits@10={t['hits@10']:.3f} "
                      f"MRR={t['mrr']:.3f}")
        results["by_mode"][mode] = {
            m: {"agg": aggregate(per_model[m]), "runs": per_model[m]}
            for m in args.models
        }

    print("\n" + "=" * 64)
    print(f"SUMMARY (test, mean +/- std)  [features={args.feature_mode}, "
          f"neg={args.neg_strategy}]")
    for mode, mr in results["by_mode"].items():
        print(f"\n[{mode}]")
        for m, d in mr.items():
            a = d["agg"]
            print(f"  {m:9s} AUROC={a['auroc']['mean']:.3f}+/-{a['auroc']['std']:.2f} "
                  f"AUPRC={a['auprc']['mean']:.3f} MRR={a['mrr']['mean']:.3f}")

    path = os.path.join(RESULTS_DIR, f"real_comparison_{args.feature_mode}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
