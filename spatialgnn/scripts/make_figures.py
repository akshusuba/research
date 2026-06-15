#!/usr/bin/env python
"""Generate publication-quality figures for the SpatialGNN proposal.

Reads the saved result JSONs in ``results/`` and produces:

  * ``fig1_real_comparison.png`` -- headline: GNN vs XGBoost/MLP across two
    tissues / technologies (LIBD DLPFC Visium, osmFISH single-cell FISH),
    showing both GraphSAGE and GAT encoders on identical expression features.
  * ``fig2_real_ablation.png``   -- the falsification test on BOTH tissues:
    intact spatial graph vs degree-matched shuffled / empty graph, against the
    blind MLP reference. The collapse proves the gain is spatial.
  * ``fig3_synthetic.png``       -- controlled mechanism benchmark (supplementary).

Run after the experiments:
    python scripts/run_real.py ...                 # -> real_comparison.json
    python scripts/run_real.py ... --out real_comparison_osmfish.json
    python scripts/run_real_ablation.py ...        # -> real_graph_ablation.json
    python scripts/run_real_ablation.py ... --out real_graph_ablation_osmfish.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatialgnn.config import FIGURES_DIR, RESULTS_DIR

# Consistent palette / labels across all figures.
MODEL_ORDER = ["xgboost", "mlp", "sage", "gat"]
COLORS = {"xgboost": "#9aa0a6", "mlp": "#5b8ff9", "sage": "#e8684a", "gat": "#7b4fae"}
LABELS = {
    "xgboost": "XGBoost (expr. only)",
    "mlp": "MLP (expr. only)",
    "sage": "Spatial GNN — GraphSAGE",
    "gat": "Spatial GNN — GAT",
}
ABL_COLORS = {"intact": "#e8684a", "shuffled": "#f6bd16", "empty": "#cfcfcf"}


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _val(res_by_model, model, metric):
    """Return (mean, std) for a model/metric from a real_comparison JSON."""
    if model not in res_by_model:
        return None, None
    m = res_by_model[model][metric]
    return float(m["mean"]), float(m.get("std", 0.0))


# --------------------------------------------------------------------------- #
# Figure 1: real-data comparison across two tissues
# --------------------------------------------------------------------------- #
def fig_real_comparison(out):
    panels = []
    dlpfc = _load("real_comparison.json")
    osm = _load("real_comparison_osmfish.json")
    if dlpfc:
        panels.append(("Human DLPFC cortex\n(10x Visium · 47,329 spots · cross-section)",
                       dlpfc["by_model"]))
    if osm:
        panels.append(("Mouse cortex\n(osmFISH · 4,839 cells · 33 genes · within-section)",
                       osm["by_model"]))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2),
                             squeeze=False)
    axes = axes[0]
    metrics = [("macro_f1", "Macro-F1"), ("ari", "ARI")]
    for ax, (title, by_model) in zip(axes, panels):
        models = [m for m in MODEL_ORDER if m in by_model]
        x = np.arange(len(metrics))
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            means = [_val(by_model, m, mk)[0] for mk, _ in metrics]
            stds = [_val(by_model, m, mk)[1] for mk, _ in metrics]
            bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                          label=LABELS.get(m, m), color=COLORS.get(m),
                          edgecolor="white", linewidth=0.6)
            for b, v in zip(bars, means):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([lbl for _, lbl in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Test score (higher is better)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", ls=":", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("Spatial GNNs map tissue domains far better than expression-only models",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 2: graph-removal ablation (falsification) on both tissues
# --------------------------------------------------------------------------- #
def fig_real_ablation(out):
    panels = []
    dlpfc = _load("real_graph_ablation.json")
    osm = _load("real_graph_ablation_osmfish.json")
    if dlpfc:
        panels.append(("Human DLPFC cortex (Visium)", dlpfc))
    if osm:
        panels.append(("Mouse cortex (osmFISH)", osm))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.0),
                             squeeze=False)
    axes = axes[0]
    conds = ["intact", "shuffled", "empty"]
    cond_lbl = {"intact": "GNN\nintact graph", "shuffled": "GNN\nshuffled edges",
                "empty": "GNN\nempty graph"}
    for ax, (title, res) in zip(axes, panels):
        means = [res["conditions"][c]["macro_f1"]["mean"] for c in conds]
        stds = [res["conditions"][c]["macro_f1"]["std"] for c in conds]
        mlp = res["mlp_reference"]["macro_f1"]["mean"]
        x = np.arange(len(conds))
        bars = ax.bar(x, means, 0.6, yerr=stds, capsize=4,
                      color=[ABL_COLORS[c] for c in conds],
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.axhline(mlp, ls="--", c="#333", lw=1.6, label=f"MLP (blind) = {mlp:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([cond_lbl[c] for c in conds])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Test macro-F1")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Falsification test: destroy the spatial graph and the GNN advantage vanishes",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 3: synthetic mechanism benchmark (supplementary)
# --------------------------------------------------------------------------- #
def fig_synthetic(out):
    res = _load("synthetic_comparison.json")
    if not res:
        return
    modes = list(res["by_mode"].keys())
    models = res["config"]["models"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(modes))
    for i, m in enumerate(models):
        means = [res["by_mode"][md][m]["agg"]["macro_f1"]["mean"] for md in modes]
        stds = [res["by_mode"][md][m]["agg"]["macro_f1"]["std"] for md in modes]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=LABELS.get(m, m), color=COLORS.get(m, "#888"),
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace("_", "-") for m in modes])
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Controlled benchmark: when features carry no domain signal,\n"
                 "only the spatial graph recovers the domain")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def main():
    fig_real_comparison(os.path.join(FIGURES_DIR, "fig1_real_comparison.png"))
    fig_real_ablation(os.path.join(FIGURES_DIR, "fig2_real_ablation.png"))
    fig_synthetic(os.path.join(FIGURES_DIR, "fig3_synthetic.png"))


if __name__ == "__main__":
    main()
