"""Canonical OncoRepurpose-GNN experiment.

Runs the 4-model x 3-regime comparison over multiple seeds, plus topology and
relation ablations, and writes one source-of-truth results JSON + a markdown
table + headline/ablation figures.

Models:  GNN (ours) | FeatureMLP | DistMult-KGE | tuned XGBoost
Regimes: transductive | inductive cold-disease (oncology) | inductive cold-drug

Usage:
  python scripts/run_experiment.py            # full run (5 seeds, real ST features)
  python scripts/run_experiment.py --smoke    # fast (1 seed, hashing features)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from oncorepurpose.config import (
    DISEASE_TYPE, DRUG_TYPE, FIGURES_DIR, RESULTS_DIR, DEFAULT_SEEDS,
)
from oncorepurpose.datasets import graph_summary, load_primekg
from oncorepurpose.evaluation.splits import (
    ablate_topology, drop_relations, make_split,
)
from oncorepurpose.evaluation.statistical_tests import cohens_d, paired_ttest
from oncorepurpose.evaluation.trainer import (
    evaluate_model, set_all_seeds, train_gnn, train_kge, train_mlp,
)
from oncorepurpose.models import DistMultKGE, FeatureMLP, HeteroGNN

REGIMES = ["transductive", "inductive_cold_dst", "inductive_cold_src"]
REGIME_LABEL = {
    "transductive": "Transductive",
    "inductive_cold_dst": "Inductive (cold-disease, oncology)",
    "inductive_cold_src": "Inductive (cold-drug)",
}
MODELS = ["GNN", "XGBoost", "MLP", "KGE"]
# AUROC/AUPRC/F1 are the trustworthy metrics for this pooled pos/neg setup.
# Pooled Hits@K / MRR are NOT meaningful here (they need per-query candidate
# ranking) and are reported separately by the deliverable, not in this table.
REPORT = ["auroc", "auprc", "f1"]


def build_gnn(data, base, in_dims, hidden, layers, dropout):
    return HeteroGNN(list(data.node_types), list(base.edge_types), in_dims,
                     hidden=hidden, num_layers=layers, dropout=dropout)


def train_eval_one(model_name, split, data, in_dims, dev, cfg, seed):
    from oncorepurpose.baselines.xgboost_baseline import run_xgboost
    set_all_seeds(seed)
    if model_name == "GNN":
        m = build_gnn(data, split.base, in_dims, cfg["hidden"], cfg["layers"], cfg["dropout"])
        m = train_gnn(m, split, dev, epochs=cfg["gnn_epochs"], patience=cfg["patience"])
        return evaluate_model(m, split, dev)
    if model_name == "MLP":
        m = FeatureMLP(list(data.node_types), in_dims, hidden=cfg["hidden"], dropout=cfg["dropout"])
        m = train_mlp(m, split, dev, epochs=cfg["mlp_epochs"], patience=cfg["patience"])
        return evaluate_model(m, split, dev)
    if model_name == "KGE":
        m = DistMultKGE(DRUG_TYPE, DISEASE_TYPE, int(data[DRUG_TYPE].num_nodes),
                        int(data[DISEASE_TYPE].num_nodes), dim=cfg["hidden"])
        m = train_kge(m, split, dev, epochs=cfg["kge_epochs"], patience=cfg["patience"])
        return evaluate_model(m, split, dev)
    if model_name == "XGBoost":
        return run_xgboost(split, data, seed=seed, n_estimators=cfg["xgb_estimators"],
                           tune=cfg["xgb_tune"], n_trials=cfg["xgb_trials"])
    raise ValueError(model_name)


def aggregate(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "values": [float(v) for v in a]}


def run(cfg) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, targets = load_primekg(with_features=True, force_fallback_features=cfg["fallback"])
    target = targets["indication"]
    print(graph_summary(data, targets))
    print(f"Target: {target} | device: {dev} | seeds: {cfg['seeds']}")
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}

    # per_seed[regime][model][seed] = metrics
    per_seed = {r: {m: {} for m in MODELS} for r in REGIMES}
    split_info = {r: {} for r in REGIMES}

    for regime in REGIMES:
        print(f"\n{'='*60}\nREGIME: {regime}\n{'='*60}")
        for seed in cfg["seeds"]:
            kw = {"restrict_oncology": True} if regime == "inductive_cold_dst" else {}
            split = make_split(data, target, regime, seed=seed, **kw)
            split_info[regime][seed] = split.info
            for model_name in MODELS:
                metrics = train_eval_one(model_name, split, data, in_dims, dev, cfg, seed)
                per_seed[regime][model_name][seed] = metrics
                print(f"  seed {seed} {model_name:8s} auroc={metrics['auroc']:.4f} auprc={metrics['auprc']:.4f}")

    results = {}
    for regime in REGIMES:
        results[regime] = {}
        for model_name in MODELS:
            vv = {}
            for seed in cfg["seeds"]:
                for k, v in per_seed[regime][model_name][seed].items():
                    vv.setdefault(k, []).append(v)
            results[regime][model_name] = {k: aggregate(v) for k, v in vv.items()}

    # Paired tests: GNN vs each baseline, per regime, AUROC.
    stats = {}
    for regime in REGIMES:
        stats[regime] = {}
        gnn_v = results[regime]["GNN"]["auroc"]["values"]
        for b in ["XGBoost", "MLP", "KGE"]:
            bv = results[regime][b]["auroc"]["values"]
            tt = paired_ttest(gnn_v, bv, alternative="greater")
            stats[regime][f"GNN_vs_{b}"] = {
                "mean_diff": float(np.mean(gnn_v) - np.mean(bv)),
                "p_value": tt["p_value"], "cohens_d": cohens_d(gnn_v, bv),
            }

    # Ablations on the regime where topology should matter most (cold-disease).
    ablations = run_ablations(data, target, in_dims, dev, cfg)

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": "PrimeKG", "target_edge_type": list(target),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "graph": graph_summary(data, targets),
        "results": results,
        "per_seed": {r: {m: {str(s): v for s, v in per_seed[r][m].items()} for m in MODELS} for r in REGIMES},
        "splits": {r: {str(s): i for s, i in split_info[r].items()} for r in REGIMES},
        "stats": stats,
        "ablations": ablations,
        "device": str(dev),
    }


def run_ablations(data, target, in_dims, dev, cfg) -> dict:
    print(f"\n{'='*60}\nABLATIONS (regime: inductive_cold_dst)\n{'='*60}")
    seeds = cfg["ablation_seeds"]
    out = {"topology": {}, "relation": {}}

    # Topology: intact vs shuffled vs empty (GNN).
    for mode in ["intact", "shuffle", "empty"]:
        vals = []
        for seed in seeds:
            split = make_split(data, target, "inductive_cold_dst", seed=seed, restrict_oncology=True)
            if mode == "intact":
                s = split
            else:
                s = ablate_topology(split, mode, seed=seed)
            set_all_seeds(seed)
            m = build_gnn(data, s.base, in_dims, cfg["hidden"], cfg["layers"], cfg["dropout"])
            m = train_gnn(m, s, dev, epochs=cfg["gnn_epochs"], patience=cfg["patience"])
            vals.append(evaluate_model(m, s, dev)["auroc"])
        out["topology"][mode] = aggregate(vals)
        print(f"  topology[{mode}] auroc={out['topology'][mode]['mean']:.4f}")

    # Relation: drop key multi-hop relation families (GNN).
    rel_groups = {
        "drop_drug_protein": ["drug_protein", "carrier", "enzyme", "target", "transporter"],
        "drop_disease_protein": ["disease_protein"],
        "drop_pathway": ["pathway"],
    }
    for name, subs in rel_groups.items():
        vals = []
        for seed in seeds:
            split = make_split(data, target, "inductive_cold_dst", seed=seed, restrict_oncology=True)
            s = drop_relations(split, subs)
            set_all_seeds(seed)
            m = build_gnn(data, s.base, in_dims, cfg["hidden"], cfg["layers"], cfg["dropout"])
            m = train_gnn(m, s, dev, epochs=cfg["gnn_epochs"], patience=cfg["patience"])
            vals.append(evaluate_model(m, s, dev)["auroc"])
        out["relation"][name] = aggregate(vals)
        print(f"  relation[{name}] auroc={out['relation'][name]['mean']:.4f}")
    return out


def make_table(out: dict) -> str:
    lines = ["| Regime | Model | " + " | ".join(m.upper() for m in REPORT) + " |",
             "|" + "|".join(["---"] * (len(REPORT) + 2)) + "|"]
    for regime in REGIMES:
        for model_name in MODELS:
            cells = []
            for metric in REPORT:
                agg = out["results"][regime][model_name].get(metric)
                cells.append(f"{agg['mean']:.3f}±{agg['std']:.3f}" if agg else "-")
            lines.append(f"| {REGIME_LABEL[regime]} | {model_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_figures(out: dict) -> None:
    # Main: AUROC per model grouped by regime.
    x = np.arange(len(REGIMES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(MODELS):
        means = [out["results"][r][model_name]["auroc"]["mean"] for r in REGIMES]
        stds = [out["results"][r][model_name]["auroc"]["std"] for r in REGIMES]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3, label=model_name)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([REGIME_LABEL[r] for r in REGIMES], rotation=12, ha="right")
    ax.set_ylabel("Test AUROC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("OncoRepurpose-GNN: GNN vs tabular/memorization baselines (PrimeKG)")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "main_results.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "main_results.pdf", bbox_inches="tight"); plt.close(fig)

    # Ablation: topology (cold-disease).
    topo = out.get("ablations", {}).get("topology", {})
    if topo:
        modes = ["intact", "shuffle", "empty"]
        means = [topo[m]["mean"] for m in modes]; stds = [topo[m]["std"] for m in modes]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(modes, means, yerr=stds, capsize=4, color=["#4C72B0", "#DD8452", "#C44E52"])
        ax.set_ylabel("GNN test AUROC (cold-disease)"); ax.set_ylim(0.4, 1.0)
        ax.set_title("Topology ablation: does the graph drive the GNN?")
        fig.tight_layout(); fig.savefig(FIGURES_DIR / "ablation_topology.png", dpi=150, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "ablation_topology.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"Saved figures to {FIGURES_DIR}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--gnn-epochs", type=int, default=50)
    p.add_argument("--mlp-epochs", type=int, default=200)
    p.add_argument("--kge-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--xgb-tune", action="store_true")
    p.add_argument("--ablation-seeds", type=int, nargs="+", default=None)
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "oncorepurpose.json"))
    args = p.parse_args()

    if args.smoke:
        cfg = dict(seeds=[0], ablation_seeds=[0], hidden=64, layers=2, dropout=0.3,
                   gnn_epochs=20, mlp_epochs=50, kge_epochs=100, patience=8,
                   xgb_estimators=200, xgb_tune=False, xgb_trials=10, fallback=True)
        out_path = Path(args.out).with_name("oncorepurpose_smoke.json")
    else:
        cfg = dict(seeds=list(args.seeds or DEFAULT_SEEDS),
                   ablation_seeds=list(args.ablation_seeds or [0, 1, 2]),
                   hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                   gnn_epochs=args.gnn_epochs, mlp_epochs=args.mlp_epochs, kge_epochs=args.kge_epochs,
                   patience=args.patience, xgb_estimators=400,
                   # Tune XGBoost by default so the tabular baseline is genuinely strong
                   # (the earlier run left it untuned despite the "tuned" claim).
                   xgb_tune=(True or args.xgb_tune),
                   xgb_trials=8, fallback=False)
        out_path = Path(args.out)

    out = run(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    md = make_table(out)
    out_path.with_suffix(".md").write_text(md + "\n")
    make_figures(out)
    print("\n" + md)
    print("\nPaired tests (GNN > baseline, AUROC):")
    for regime in REGIMES:
        for comp, st in out["stats"][regime].items():
            print(f"  [{regime}] {comp}: d={st['mean_diff']:+.3f} p={st['p_value']:.4f}")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
