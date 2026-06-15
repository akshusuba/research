"""Survival-only re-run (the full run was killed externally mid-survival).

Same cohort construction and config as scripts/run_real_imc.py; writes
results/real_imc_survival.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from spatial_prognosis.config import ModelConfig, TrainConfig, RESULTS_DIR
from spatial_prognosis.data.real_imc import binarize_survival
from spatial_prognosis.splits import make_split
from spatial_prognosis.experiment import METRIC_KEYS
from spatial_prognosis.ablation import run_graph_ablation
from scripts.run_real_imc import load_cells, build, run_label, class_balance, SEEDS, IMC, MAX_CELLS, K_NEIGHBORS


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    cells = load_cells()
    lab = pd.read_csv(os.path.join(IMC, "basel_labels.csv"))
    lab["sample_id"] = lab["sample_id"].astype(str)

    surv = lab.dropna(subset=["OSmonth"]).copy()
    y = binarize_survival(surv["OSmonth"].to_numpy(), surv["event"].to_numpy(),
                          cutoff_months=60.0)
    surv_map = {str(s): int(v) for s, v in zip(surv["sample_id"], y) if v != -1}
    ds_surv = build(cells, surv_map)
    print(f"SURVIVAL cohort: {len(ds_surv.graphs)} patients, "
          f"balance={class_balance(surv_map)} (1=survived>5yr, 0=died<5yr)",
          flush=True)
    surv_results = run_label("survival5y", ds_surv, model_cfg, train_cfg)

    print("\n--- graph-shuffle ablation (survival5y) ---", flush=True)
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

    print("\n" + "=" * 64)
    print(f"SUMMARY 5-YEAR SURVIVAL  (test, mean +/- std over {len(SEEDS)} seeds)")
    print("=" * 64)
    print(f"{'model':9s} " + " ".join(f"{k:>16s}" for k in METRIC_KEYS))
    for m, d in surv_results.items():
        a = d["agg"]
        print(f"{m:9s} " + " ".join(
            f"{a[k]['mean']:.3f}+/-{a[k]['std']:.2f}" for k in METRIC_KEYS))
    ab = surv_abl
    print("ablation (GNN test macro-F1): " + " ".join(
        f"{c}={ab['conditions'][c]['macro_f1']['mean']:.3f}"
        for c in ["intact", "shuffled", "empty"])
        + f"  xgb_ref={ab['composition_reference']['macro_f1']['mean']:.3f}")
    print("saved:", os.path.join(RESULTS_DIR, "real_imc_survival.json"), flush=True)


if __name__ == "__main__":
    main()
