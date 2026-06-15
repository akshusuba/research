#!/usr/bin/env python
"""Run the composition-vs-arrangement comparison on a real IMC/spatial cohort.

Provide a long-form single-cell CSV (one row per cell, with a sample id, x, y,
optional cell-type column, and marker columns) plus a clinical CSV mapping each
sample to an outcome label (e.g., tumor grade). Builds per-patient spatial cell
graphs and compares composition-only baselines vs the spatial GNN.

Example (column names depend on the dataset):
  python scripts/run_real.py \
      --cells data/imc/cells.csv --clinical data/imc/clinical.csv \
      --sample-col core --x-col Location_Center_X --y-col Location_Center_Y \
      --celltype-col cluster --label-col grade
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_prognosis.config import RESULTS_DIR, ModelConfig, SplitConfig, TrainConfig
from spatial_prognosis.data.real_imc import build_cohort_from_tables
from spatial_prognosis.data.synthetic import summarize
from spatial_prognosis.experiment import aggregate
from spatial_prognosis.splits import make_split
from spatial_prognosis.train import train_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--sample-col", required=True)
    ap.add_argument("--x-col", required=True)
    ap.add_argument("--y-col", required=True)
    ap.add_argument("--celltype-col", default=None)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--clinical-sample-col", default=None,
                    help="sample id column in clinical CSV (defaults to --sample-col)")
    ap.add_argument("--models", nargs="+", default=["logreg", "xgboost", "mlp", "sage"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-cells", type=int, default=1500)
    args = ap.parse_args()

    cells = pd.read_csv(args.cells)
    clinical = pd.read_csv(args.clinical)
    csample = args.clinical_sample_col or args.sample_col
    clinical = clinical.dropna(subset=[args.label_col])
    labels_raw = dict(zip(clinical[csample].astype(str), clinical[args.label_col]))
    # encode labels to 0..C-1
    classes = sorted(set(labels_raw.values()))
    enc = {c: i for i, c in enumerate(classes)}
    labels = {k: enc[v] for k, v in labels_raw.items()}
    print(f"Label classes: {classes}")

    # optional per-patient cell cap for CPU memory
    if args.max_cells:
        cells = (cells.groupby(args.sample_col, group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), args.max_cells), random_state=0)))

    ds = build_cohort_from_tables(
        cells, labels, sample_col=args.sample_col, x_col=args.x_col,
        y_col=args.y_col, celltype_col=args.celltype_col, k_neighbors=args.k)
    print("Cohort:", json.dumps(summarize(ds)))

    mcfg = ModelConfig(encoder="sage")
    tcfg = TrainConfig(epochs=150, patience=30)
    per_model = {m: [] for m in args.models}
    for seed in args.seeds:
        split = make_split(ds, SplitConfig(seed=seed))
        for m in args.models:
            run = train_any(m, ds, split, mcfg, tcfg, seed=seed)
            per_model[m].append(run)
            t = run["test"]
            print(f"  seed={seed} {m:8s} acc={t['accuracy']:.3f} "
                  f"F1={t['macro_f1']:.3f} AUROC={t['auroc']:.3f}")

    results = {"cohort": summarize(ds), "label": args.label_col,
               "by_model": {m: aggregate(per_model[m]) for m in args.models}}
    print("\nSUMMARY (test, mean +/- std)")
    for m, a in results["by_model"].items():
        print(f"  {m:8s} F1={a['macro_f1']['mean']:.3f}+/-{a['macro_f1']['std']:.2f} "
              f"AUROC={a['auroc']['mean']:.3f}")
    path = os.path.join(RESULTS_DIR, f"real_imc_{args.label_col}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
