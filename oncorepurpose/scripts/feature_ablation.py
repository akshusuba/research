"""DECISIVE feature ablation: GNN vs tuned XGBoost vs MLP in the inductive
cold-disease (oncology) regime, under NON-SEMANTIC node features.

Question
--------
Does a heterogeneous GNN out-rank a tuned XGBoost in the inductive cold-disease
regime when node features carry NO name/text semantics -- i.e. when XGBoost can no
longer exploit name-embedding similarity?

We compare {GNN, XGBoost(tuned), MLP} in ONE regime (`inductive_cold_dst`,
restrict_oncology=True) under three feature settings, all sharing the SAME features
between every model:
  (a) text       -- existing SentenceTransformer name embeddings (sanity; expect GNN<XGB)
  (b) structural -- build_structural_features: local graph structure, no semantics (THE TEST)
  (c) random     -- fixed random vector per node (REFERENCE ONLY: cripples XGBoost to
                    ~chance, isolates a pure-structure GNN; NOT a fair comparison)

Reports a clean AUROC table (rows=feature setting, cols=GNN/XGBoost/MLP, mean over
seeds) and writes results/feature_ablation.json.

Usage:
  PYTHONPATH=. python scripts/feature_ablation.py            # full: 2 seeds
  PYTHONPATH=. python scripts/feature_ablation.py --smoke    # 1 seed, reduced epochs
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import torch

from oncorepurpose.baselines.xgboost_baseline import run_xgboost
from oncorepurpose.datasets import graph_summary, load_primekg
from oncorepurpose.evaluation.splits import make_split
from oncorepurpose.evaluation.trainer import (
    evaluate_model, set_all_seeds, train_gnn, train_mlp,
)
from oncorepurpose.models import FeatureMLP, HeteroGNN
from oncorepurpose.struct_features import (
    build_random_features, build_structural_features,
)
from oncorepurpose.config import RESULTS_DIR

REGIME = "inductive_cold_dst"
SETTINGS = ["text", "structural", "random"]
MODELS = ["GNN", "XGBoost", "MLP"]


def apply_features(data, feats):
    """Point every data[nt].x at the chosen feature tensor for this setting."""
    for nt in data.node_types:
        data[nt].x = feats[nt]


def train_eval_one(model_name, split, data, in_dims, dev, cfg, seed):
    set_all_seeds(seed)
    if model_name == "GNN":
        m = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims,
                      hidden=cfg["hidden"], num_layers=cfg["layers"], dropout=cfg["dropout"])
        m = train_gnn(m, split, dev, epochs=cfg["gnn_epochs"], patience=cfg["patience"])
        return evaluate_model(m, split, dev)
    if model_name == "MLP":
        m = FeatureMLP(list(data.node_types), in_dims, hidden=cfg["hidden"], dropout=cfg["dropout"])
        m = train_mlp(m, split, dev, epochs=cfg["mlp_epochs"], patience=cfg["patience"])
        return evaluate_model(m, split, dev)
    if model_name == "XGBoost":
        return run_xgboost(split, data, seed=seed, n_estimators=cfg["xgb_estimators"],
                           tune=cfg["xgb_tune"], n_trials=cfg["xgb_trials"])
    raise ValueError(model_name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    args = p.parse_args()

    if args.smoke:
        cfg = dict(seeds=[0], hidden=64, layers=2, dropout=0.3,
                   gnn_epochs=10, mlp_epochs=40, patience=6,
                   xgb_estimators=200, xgb_tune=True, xgb_trials=2,
                   random_dim=64)
        out_name = "feature_ablation_smoke.json"
    else:
        cfg = dict(seeds=list(args.seeds or [0, 1]), hidden=128, layers=2, dropout=0.3,
                   gnn_epochs=50, mlp_epochs=200, patience=10,
                   xgb_estimators=400, xgb_tune=True, xgb_trials=8,
                   random_dim=64)
        out_name = "feature_ablation.json"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} | seeds: {cfg['seeds']} | regime: {REGIME} (oncology)")

    data, targets = load_primekg(with_features=True)
    target = targets["indication"]
    print(graph_summary(data, targets))

    # Pre-build the three feature sets ONCE. Keep the original text features safe by
    # cloning before any in-place builder runs.
    text_feats = {nt: data[nt].x.clone() for nt in data.node_types}
    struct_feats = build_structural_features(data, assign=False)
    random_feats = build_random_features(data, dim=cfg["random_dim"], seed=12345, assign=False)
    feat_sets = {"text": text_feats, "structural": struct_feats, "random": random_feats}
    feat_dims = {s: int(next(iter(f.values())).size(1)) for s, f in feat_sets.items()}
    print("feature dims:", feat_dims)

    # per_seed[setting][model][seed] = auroc
    per_seed = {s: {m: {} for m in MODELS} for s in SETTINGS}
    split_info = {}

    for setting in SETTINGS:
        print(f"\n{'='*64}\nFEATURE SETTING: {setting}  (dim={feat_dims[setting]})\n{'='*64}")
        apply_features(data, feat_sets[setting])
        in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}
        for seed in cfg["seeds"]:
            split = make_split(data, target, REGIME, seed=seed, restrict_oncology=True)
            split_info[seed] = split.info
            for model_name in MODELS:
                metrics = train_eval_one(model_name, split, data, in_dims, dev, cfg, seed)
                per_seed[setting][model_name][seed] = float(metrics["auroc"])
                print(f"  [{setting}] seed {seed} {model_name:8s} auroc={metrics['auroc']:.4f}")

    # Aggregate: mean AUROC over seeds.
    grid = {s: {m: float(np.mean(list(per_seed[s][m].values()))) for m in MODELS} for s in SETTINGS}

    # Table
    header = f"{'feature':<12} | " + " | ".join(f"{m:>9}" for m in MODELS)
    lines = ["", "AUROC (mean over seeds) -- regime: inductive cold-disease (oncology)",
             header, "-" * len(header)]
    for s in SETTINGS:
        row = f"{s:<12} | " + " | ".join(f"{grid[s][m]:>9.4f}" for m in MODELS)
        lines.append(row)
    table = "\n".join(lines)
    print(table)

    # Interpretation: GNN vs XGBoost under STRUCTURAL features (the decisive test).
    g = grid["structural"]["GNN"]
    x = grid["structural"]["XGBoost"]
    margin = g - x
    gnn_wins = margin > 0
    interp = (
        f"Under STRUCTURAL (non-semantic) features in the cold-disease (oncology) "
        f"regime, GNN AUROC={g:.4f} vs XGBoost AUROC={x:.4f} "
        f"(margin {margin:+.4f}). GNN {'BEATS' if gnn_wins else 'does NOT beat'} "
        f"tuned XGBoost under structural features."
    )
    print("\n" + interp)

    out = {
        "timestamp": datetime.now().isoformat(),
        "regime": REGIME,
        "restrict_oncology": True,
        "target_edge_type": list(target),
        "seeds": cfg["seeds"],
        "config": cfg,
        "feature_dims": feat_dims,
        "auroc_grid_mean": grid,
        "per_seed_auroc": {s: {m: {str(k): v for k, v in per_seed[s][m].items()} for m in MODELS}
                           for s in SETTINGS},
        "split_info": {str(k): v for k, v in split_info.items()},
        "structural_features": (
            "log1p(total_degree) + log1p(per-relation incident counts) for top-12 "
            "global non-therapeutic relations (z-scored) + one-hot(node_type); "
            "drug<->disease therapeutic edges excluded to avoid target leakage; "
            "identical construction/dimensionality across node types."
        ),
        "random_features_note": (
            "Fixed random vector per node (dim={}); REFERENCE ONLY -- not a fair "
            "comparison (XGBoost has no signal -> ~chance).".format(cfg["random_dim"])
        ),
        "key_question": (
            "Under STRUCTURAL (non-semantic) features in cold-disease, does GNN "
            "AUROC exceed XGBoost AUROC?"
        ),
        "gnn_beats_xgboost_structural": bool(gnn_wins),
        "structural_margin_gnn_minus_xgb": float(margin),
        "interpretation": interp,
    }
    out_path = RESULTS_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
