"""Honest real-data test: does spatial ARRANGEMENT beat composition on the
Jackson-Fischer 2020 Basel breast-cancer IMC cohort?

Builds per-core spatial cell graphs (nodes = cells, features = one-hot cell type
from marker-derived metaclusters; edges = spatial kNN) and compares the spatial
GNN against composition-only baselines (LogReg / XGBoost / MLP on cell-type
proportions). Runs the graph-shuffle ablation as the falsification test.

NOTE on labels: the provided trainer/metrics are binary (predict_proba[:,1] >=
0.5; AUROC only for 2 classes). To keep the comparison fair we use BINARY
outcomes:
  * grade_bin : high grade (G3) vs low grade (G1/G2)
  * survival  : 5-year overall survival via binarize_survival
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from spatial_prognosis.config import ModelConfig, SplitConfig, TrainConfig, RESULTS_DIR
from spatial_prognosis.data.real_imc import build_cohort_from_tables, binarize_survival
from spatial_prognosis.splits import make_split
from spatial_prognosis.train import train_any
from spatial_prognosis.experiment import aggregate, METRIC_KEYS
from spatial_prognosis.ablation import run_graph_ablation

IMC = os.path.join(HERE, "..", "data", "imc")
MAX_CELLS = 1500
K_NEIGHBORS = 6
MIN_CELLS = 50
SEEDS = [0, 1, 2]
MODELS = ["logreg", "xgboost", "mlp", "sage"]


def load_cells(max_cells=MAX_CELLS, seed=0):
    cells = pd.read_csv(os.path.join(IMC, "basel_cells.csv.gz"))
    rng = np.random.default_rng(seed)
    parts = []
    for s, grp in cells.groupby("sample_id"):
        if len(grp) > max_cells:
            grp = grp.iloc[rng.choice(len(grp), max_cells, replace=False)]
        parts.append(grp)
    out = pd.concat(parts, ignore_index=True)
    return out


def build(cells, labels_dict):
    return build_cohort_from_tables(
        cells, labels_dict,
        sample_col="sample_id", x_col="x", y_col="y",
        celltype_col="cell_type", marker_cols=[],
        k_neighbors=K_NEIGHBORS, min_cells=MIN_CELLS,
    )


def run_label(name, ds, model_cfg, train_cfg):
    per_model = {m: [] for m in MODELS}
    for seed in SEEDS:
        split = make_split(ds, SplitConfig(seed=seed))
        for m in MODELS:
            run = train_any(m, ds, split, model_cfg, train_cfg, seed=seed)
            per_model[m].append(run)
            t = run["test"]
            print(f"  [{name}] seed={seed} {m:8s} acc={t['accuracy']:.3f} "
                  f"F1={t['macro_f1']:.3f} AUROC={t['auroc']:.3f}")
    return {m: {"agg": aggregate(per_model[m]), "runs": per_model[m]} for m in MODELS}


def class_balance(labels_dict):
    v = list(labels_dict.values())
    return {int(c): int((np.array(v) == c).sum()) for c in sorted(set(v))}


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    cells = load_cells()
    lab = pd.read_csv(os.path.join(IMC, "basel_labels.csv"))
    lab["sample_id"] = lab["sample_id"].astype(str)

    cores_with_cells = set(cells["sample_id"].unique())
    n_cells_kept = len(cells)
    n_types = cells["cell_type"].nunique()

    # ---------- label 1: grade dichotomized (G3 vs G1/G2) ----------
    grade_map = {str(r.sample_id): int(r.grade == 3)
                 for r in lab.itertuples() if not pd.isna(r.grade)}
    ds_grade = build(cells, grade_map)
    print(f"\nGRADE cohort: {len(ds_grade.graphs)} patients, "
          f"{ds_grade.num_features} feats, balance={class_balance(grade_map)}")
    grade_results = run_label("grade_bin", ds_grade, model_cfg, train_cfg)

    print("\n--- graph-shuffle ablation (grade_bin) ---")
    grade_abl = run_graph_ablation(ds_grade, make_split, model_cfg, train_cfg, SEEDS)

    out_grade = {
        "label": "grade_bin (G3 vs G1/G2)",
        "cohort": {"patients_graphs": len(ds_grade.graphs),
                   "cells_kept_total": int(n_cells_kept),
                   "cell_types": int(n_types),
                   "num_features": int(ds_grade.num_features),
                   "max_cells_per_patient": MAX_CELLS,
                   "k_neighbors": K_NEIGHBORS,
                   "class_balance": class_balance(grade_map)},
        "by_model": grade_results,
        "ablation": grade_abl,
        "seeds": SEEDS,
    }
    with open(os.path.join(RESULTS_DIR, "real_imc_grade.json"), "w") as f:
        json.dump(out_grade, f, indent=2)

    # ---------- label 2: 5-year overall survival ----------
    surv = lab.dropna(subset=["OSmonth"]).copy()
    y = binarize_survival(surv["OSmonth"].to_numpy(), surv["event"].to_numpy(),
                          cutoff_months=60.0)
    surv_map = {str(s): int(v) for s, v in zip(surv["sample_id"], y) if v != -1}
    ds_surv = build(cells, surv_map)
    print(f"\nSURVIVAL cohort: {len(ds_surv.graphs)} patients, "
          f"balance={class_balance(surv_map)} (1=survived>5yr, 0=died<5yr)")
    surv_results = run_label("survival5y", ds_surv, model_cfg, train_cfg)

    print("\n--- graph-shuffle ablation (survival5y) ---")
    surv_abl = run_graph_ablation(ds_surv, make_split, model_cfg, train_cfg, SEEDS)

    out_surv = {
        "label": "survival5y (1=survived past 60mo, 0=died before)",
        "cohort": {"patients_graphs": len(ds_surv.graphs),
                   "num_features": int(ds_surv.num_features),
                   "max_cells_per_patient": MAX_CELLS,
                   "k_neighbors": K_NEIGHBORS,
                   "class_balance": class_balance(surv_map)},
        "by_model": surv_results,
        "ablation": surv_abl,
        "seeds": SEEDS,
    }
    with open(os.path.join(RESULTS_DIR, "real_imc_survival.json"), "w") as f:
        json.dump(out_surv, f, indent=2)

    # ---------- print summary tables ----------
    for tag, res in [("GRADE (G3 vs G1/G2)", out_grade),
                     ("5-YEAR SURVIVAL", out_surv)]:
        print("\n" + "=" * 64)
        print(f"SUMMARY {tag}  (test, mean +/- std over {len(SEEDS)} seeds)")
        print("=" * 64)
        print(f"{'model':9s} " + " ".join(f"{k:>16s}" for k in METRIC_KEYS))
        for m, d in res["by_model"].items():
            a = d["agg"]
            print(f"{m:9s} " + " ".join(
                f"{a[k]['mean']:.3f}+/-{a[k]['std']:.2f}" for k in METRIC_KEYS))
        ab = res["ablation"]
        print("ablation (GNN test macro-F1): " + " ".join(
            f"{c}={ab['conditions'][c]['macro_f1']['mean']:.3f}"
            for c in ["intact", "shuffled", "empty"])
            + f"  xgb_ref={ab['composition_reference']['macro_f1']['mean']:.3f}")

    print("\nsaved:", os.path.join(RESULTS_DIR, "real_imc_grade.json"),
          "and real_imc_survival.json")


if __name__ == "__main__":
    main()
