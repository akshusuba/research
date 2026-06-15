#!/usr/bin/env python
"""Run the SpatialGNN comparison on a real spatial-omics dataset.

Provide any AnnData .h5ad with 2D coordinates (obsm['spatial']) and a domain
label column. Recommended: the LIBD DLPFC Visium benchmark (see
spatialgnn/data/real.py for the one-line R export to .h5ad).

Usage:
    python scripts/run_real.py --h5ad dlpfc.h5ad \
        --label-key layer_guess_reordered --sample-key sample_id

Cross-section evaluation (leakage-safe) is used when --sample-key is given.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialgnn.config import (RESULTS_DIR, ModelConfig, SplitConfig,
                               TrainConfig)
from spatialgnn.data.real import load_h5ad_spatial
from spatialgnn.data.synthetic import summarize
from spatialgnn.experiment import aggregate
from spatialgnn.splits import make_split
from spatialgnn.train import train_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--label-key", required=True)
    ap.add_argument("--spatial-key", default="spatial")
    ap.add_argument("--sample-key", default=None)
    ap.add_argument("--models", nargs="+", default=["xgboost", "mlp", "sage", "gat"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--mode", default=None, help="cross_slice|within_slice|stratified")
    ap.add_argument("--out", default="real_comparison.json")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading", args.h5ad)
    ds = load_h5ad_spatial(args.h5ad, label_key=args.label_key,
                           spatial_key=args.spatial_key,
                           sample_key=args.sample_key, k_neighbors=args.k)
    print("Stats:", json.dumps(summarize(ds), indent=2))

    mode = args.mode or ("cross_slice" if args.sample_key else "within_slice")
    mcfg = ModelConfig(encoder="sage")
    tcfg = TrainConfig(epochs=args.epochs, patience=40, device=device)

    print(f"\n=== {mode} (device={device}) ===")
    per_model = {m: [] for m in args.models}
    for seed in args.seeds:
        split = make_split(ds, SplitConfig(mode=mode, seed=seed))
        for m in args.models:
            run = train_any(m, split, mcfg, tcfg, seed=seed)
            per_model[m].append(run)
            t = run["test"]
            print(f"  seed={seed} {m:8s} acc={t['accuracy']:.3f} "
                  f"F1={t['macro_f1']:.3f} ARI={t['ari']:.3f}")

    results = {"stats": summarize(ds), "mode": mode,
               "by_model": {m: aggregate(per_model[m]) for m in args.models}}
    print("\nSUMMARY (test, mean +/- std)")
    for m, a in results["by_model"].items():
        print(f"  {m:8s} acc={a['accuracy']['mean']:.3f} "
              f"F1={a['macro_f1']['mean']:.3f}+/-{a['macro_f1']['std']:.2f} "
              f"ARI={a['ari']['mean']:.3f}")

    path = os.path.join(RESULTS_DIR, args.out)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
