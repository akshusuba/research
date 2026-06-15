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
from spatial_prognosis.config import FIGURES_DIR, RESULTS_DIR

LABELS = {"logreg": "LogReg (composition)", "xgboost": "XGBoost (composition)",
          "mlp": "MLP (composition)", "sage": "Spatial GNN (ours)"}
COLORS = {"logreg": "#b0b0b0", "xgboost": "#9e9e9e", "mlp": "#5b8ff9",
          "sage": "#e8684a"}


def fig_comparison(path, out):
    with open(path) as f:
        res = json.load(f)
    models = res["config"]["models"]
    means = [res["by_model"][m]["agg"]["macro_f1"]["mean"] for m in models]
    stds = [res["by_model"][m]["agg"]["macro_f1"]["std"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    ax.bar(x, means, 0.6, yerr=stds, capsize=4, color=[COLORS.get(m) for m in models])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in models], rotation=12, ha="right")
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls=":", c="k", alpha=0.5, lw=1, label="chance")
    ax.set_title("Outcome from tumor arrangement: composition is blind, the GNN isn't")
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def fig_ablation(path, out):
    with open(path) as f:
        res = json.load(f)
    conds = list(res["conditions"].keys())
    means = [res["conditions"][c]["macro_f1"]["mean"] for c in conds]
    stds = [res["conditions"][c]["macro_f1"]["std"] for c in conds]
    comp = res["composition_reference"]["macro_f1"]["mean"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conds))
    ax.bar(x, means, 0.6, yerr=stds, capsize=4,
           color=["#e8684a", "#f6bd16", "#cccccc"][:len(conds)])
    ax.axhline(comp, ls="--", c="#444", lw=1.5, label=f"XGBoost (composition) = {comp:.2f}")
    ax.set_xticks(x); ax.set_xticklabels([f"GNN\n({c})" for c in conds])
    ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Graph-shuffle ablation:\nthe GNN's advantage is the spatial arrangement")
    ax.legend(loc="center right", fontsize=9)
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
