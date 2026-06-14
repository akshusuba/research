#!/usr/bin/env python
"""Generate figures from saved synthetic results."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatialgnn.config import FIGURES_DIR, RESULTS_DIR

COLORS = {"xgboost": "#9e9e9e", "mlp": "#5b8ff9", "sage": "#e8684a"}
LABELS = {"xgboost": "XGBoost", "mlp": "MLP (features only)", "sage": "Spatial GNN (ours)"}


def fig_comparison(path, out):
    with open(path) as f:
        res = json.load(f)
    modes = list(res["by_mode"].keys())
    models = res["config"]["models"]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(modes))
    for i, m in enumerate(models):
        means = [res["by_mode"][md][m]["agg"]["macro_f1"]["mean"] for md in modes]
        stds = [res["by_mode"][md][m]["agg"]["macro_f1"]["std"] for md in modes]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=LABELS.get(m, m), color=COLORS.get(m))
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", "-") for m in modes])
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Spatial domain classification: the GNN earns its keep")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def fig_ablation(path, out):
    with open(path) as f:
        res = json.load(f)
    conds = list(res["conditions"].keys())
    means = [res["conditions"][c]["macro_f1"]["mean"] for c in conds]
    stds = [res["conditions"][c]["macro_f1"]["std"] for c in conds]
    mlp = res["mlp_reference"]["macro_f1"]["mean"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conds))
    ax.bar(x, means, 0.6, yerr=stds, capsize=4,
           color=["#e8684a", "#f6bd16", "#cccccc"][:len(conds)])
    ax.axhline(mlp, ls="--", c="#444", lw=1.5, label=f"MLP (blind) = {mlp:.2f}")
    ax.set_xticks(x); ax.set_xticklabels([f"GNN\n({c})" for c in conds])
    ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, 1.0)
    ax.set_title("Graph-removal ablation:\nthe GNN's advantage is spatial")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def main():
    comp = os.path.join(RESULTS_DIR, "synthetic_comparison.json")
    abl = os.path.join(RESULTS_DIR, "graph_ablation.json")
    if os.path.exists(comp):
        fig_comparison(comp, os.path.join(FIGURES_DIR, "fig1_comparison.png"))
    if os.path.exists(abl):
        fig_ablation(abl, os.path.join(FIGURES_DIR, "fig2_graph_ablation.png"))


if __name__ == "__main__":
    main()
