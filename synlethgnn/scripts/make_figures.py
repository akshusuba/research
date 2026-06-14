#!/usr/bin/env python
"""Generate publication figures from saved results JSON.

Produces:
  * fig1_comparison.png  -- AUROC by model, grouped by split regime (the
    headline "GNN shines inductively" plot).
  * fig2_topology_ablation.png -- GNN AUPRC under intact/rewired/empty graphs
    vs the structure-blind MLP (the falsification test).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synlethgnn.config import FIGURES_DIR, RESULTS_DIR

MODEL_COLORS = {"mlp": "#9e9e9e", "node2vec": "#5b8ff9", "gnn": "#e8684a"}
MODEL_LABELS = {"mlp": "MLP (features only)", "node2vec": "node2vec",
                "gnn": "GNN (ours)"}


def fig_comparison(comp_path: str, out: str):
    with open(comp_path) as f:
        res = json.load(f)
    modes = list(res["by_mode"].keys())
    models = res["config"]["models"]
    metric = "auroc"

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(modes))
    for i, m in enumerate(models):
        means = [res["by_mode"][mode][m]["agg"][metric]["mean"] for mode in modes]
        stds = [res["by_mode"][mode][m]["agg"][metric]["std"] for mode in modes]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=MODEL_LABELS.get(m, m), color=MODEL_COLORS.get(m))
    ax.axhline(0.5, ls="--", c="k", lw=1, alpha=0.6, label="chance")
    ax.set_xticks(x + width)
    ax.set_xticklabels([mode.capitalize() for mode in modes])
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Synthetic lethality: where the GNN earns its keep")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_ablation(abl_path: str, out: str):
    with open(abl_path) as f:
        res = json.load(f)
    conds = list(res["conditions"].keys())
    means = [res["conditions"][c]["auprc"]["mean"] for c in conds]
    stds = [res["conditions"][c]["auprc"]["std"] for c in conds]
    mlp_mean = res["mlp_reference"]["auprc"]["mean"]

    labels = [f"GNN\n({c})" for c in conds]
    colors = ["#e8684a", "#f6bd16", "#cccccc"][:len(conds)]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conds))
    ax.bar(x, means, 0.6, yerr=stds, capsize=4, color=colors)
    ax.axhline(mlp_mean, ls="--", c="#444", lw=1.5,
               label=f"MLP (blind) = {mlp_mean:.2f}")
    ax.axhline(0.5, ls=":", c="k", lw=1, alpha=0.6, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test AUPRC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Topology-removal ablation:\nthe GNN's advantage is structural")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    comp = os.path.join(RESULTS_DIR, "synthetic_comparison.json")
    abl = os.path.join(RESULTS_DIR, "topology_ablation.json")
    if os.path.exists(comp):
        fig_comparison(comp, os.path.join(FIGURES_DIR, "fig1_comparison.png"))
    else:
        print(f"  missing {comp}")
    if os.path.exists(abl):
        fig_ablation(abl, os.path.join(FIGURES_DIR, "fig2_topology_ablation.png"))
    else:
        print(f"  missing {abl}")


if __name__ == "__main__":
    main()
