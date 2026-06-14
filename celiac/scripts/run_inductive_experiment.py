"""
Canonical inductive vs transductive experiment on PrimeKG-Celiac.

Runs a 2x2 model design (content features vs id embeddings) x (GNN vs no graph)
across two evaluation regimes (transductive random-edge holdout vs inductive
node-disjoint holdout), over multiple seeds, and writes a single source-of-truth
results file plus a headline table and figure.

The scientific question: does message passing add value beyond memorising node
identity? The transductive regime is expected to show feature/embedding and
graph/no-graph models performing similarly (structure is not required to
reconnect known nodes). The inductive regime is expected to separate them:
embedding-lookup models collapse toward chance on unseen nodes, while the
feature GNN retains performance by using content features and neighbourhood
structure.

Usage:
    python scripts/run_inductive_experiment.py            # full run (5 seeds)
    python scripts/run_inductive_experiment.py --smoke    # fast 1-seed CPU check
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from celiac.config import FIGURES_DIR, PROJECT_ROOT
from celiac.datasets import graph_summary, load_primekg
from celiac.evaluation.experiment_runner import set_all_seeds
from celiac.evaluation.metrics import compute_all_metrics
from celiac.evaluation.splits import make_split
from celiac.evaluation.statistical_tests import cohens_d, paired_ttest
from celiac.models_inductive import MODE_DISPLAY, MODES, HeteroLinkModel

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_SEEDS = [0, 1, 2, 42, 123]
REGIMES = ["transductive", "inductive"]
REPORT_METRICS = ["auroc", "auprc", "hits@10", "mrr", "f1"]


def evaluate(model: HeteroLinkModel, data, target_edge_type, device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        z = model(data)
        eli = data[target_edge_type].edge_label_index
        y = data[target_edge_type].edge_label
        scores = torch.sigmoid(model.decode(z, target_edge_type, eli))
    return compute_all_metrics(y.cpu(), scores.cpu())


def train_one(
    model: HeteroLinkModel,
    split,
    target_edge_type,
    epochs: int,
    patience: int,
    lr: float,
    device: torch.device,
    verbose: bool = False,
) -> HeteroLinkModel:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_eli = split.train[target_edge_type].edge_label_index
    train_y = split.train[target_edge_type].edge_label.to(device)

    best_val = -1.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(split.train)
        pred = model.decode(z, target_edge_type, train_eli)
        loss = F.binary_cross_entropy_with_logits(pred, train_y)
        loss.backward()
        optimizer.step()

        val_auroc = evaluate(model, split.val, target_edge_type, device)["auroc"]
        if val_auroc > best_val:
            best_val = val_auroc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if verbose and (epoch + 1) % 20 == 0:
            print(f"      epoch {epoch+1:3d} loss={loss.item():.4f} val_auroc={val_auroc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def aggregate(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": [float(v) for v in arr]}


def run(
    seeds: List[int],
    epochs: int,
    patience: int,
    lr: float,
    hidden: int,
    num_layers: int,
    dropout: float,
    holdout_side: str,
    fallback_features: bool,
    device: torch.device,
) -> dict:
    data, target_edge_type = load_primekg(
        with_features=True, force_fallback_features=fallback_features
    )
    print(graph_summary(data, target_edge_type))

    # per_seed[regime][mode][seed] = metrics dict
    per_seed: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {
        r: {m: {} for m in MODES} for r in REGIMES
    }
    split_info: Dict[str, Dict[int, dict]] = {r: {} for r in REGIMES}

    for regime in REGIMES:
        print(f"\n{'='*64}\nREGIME: {regime}\n{'='*64}")
        for seed in seeds:
            split = make_split(
                data, target_edge_type, regime, seed=seed, holdout_side=holdout_side
            )
            split_info[regime][seed] = split.info
            print(f"  seed {seed}: {split.info}")
            for mode in MODES:
                set_all_seeds(seed)  # fair, identical init conditions per model
                model = HeteroLinkModel(
                    split.train, target_edge_type, mode=mode,
                    hidden_channels=hidden, num_layers=num_layers, dropout=dropout,
                )
                model = train_one(model, split, target_edge_type, epochs, patience, lr, device)
                metrics = evaluate(model, split.test, target_edge_type, device)
                per_seed[regime][mode][seed] = metrics
                print(f"    {mode:14s} test AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f}")

    # Aggregate
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for regime in REGIMES:
        results[regime] = {}
        for mode in MODES:
            metric_values: Dict[str, List[float]] = {}
            for seed in seeds:
                for metric, val in per_seed[regime][mode][seed].items():
                    metric_values.setdefault(metric, []).append(val)
            results[regime][mode] = {metric: aggregate(vals) for metric, vals in metric_values.items()}

    # Paired statistical tests: feature_gnn vs the other three, per regime, on AUROC.
    stats: Dict[str, dict] = {}
    ref = "feature_gnn"
    for regime in REGIMES:
        stats[regime] = {}
        ref_vals = results[regime][ref]["auroc"]["values"]
        for mode in MODES:
            if mode == ref:
                continue
            other_vals = results[regime][mode]["auroc"]["values"]
            tt = paired_ttest(ref_vals, other_vals, alternative="greater")
            stats[regime][f"{ref}_vs_{mode}"] = {
                "metric": "auroc",
                "mean_diff": float(np.mean(ref_vals) - np.mean(other_vals)),
                "t_statistic": tt["t_statistic"],
                "p_value": tt["p_value"],
                "cohens_d": float(cohens_d(ref_vals, other_vals)),
            }

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": "PrimeKG-Celiac",
        "config": {
            "seeds": seeds,
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "hidden_channels": hidden,
            "num_layers": num_layers,
            "dropout": dropout,
            "holdout_side": holdout_side,
            "feature_source": "hashing-fallback" if fallback_features else "sentence-transformers/all-MiniLM-L6-v2",
            "target_edge_type": list(target_edge_type),
            "device": str(device),
        },
        "graph": graph_summary(data, target_edge_type),
        "results": results,
        "per_seed": {r: {m: {str(s): v for s, v in per_seed[r][m].items()} for m in MODES} for r in REGIMES},
        "splits": {r: {str(s): info for s, info in split_info[r].items()} for r in REGIMES},
        "stats": stats,
    }


def make_table(out: dict, fmt: str = "markdown") -> str:
    results = out["results"]
    metrics = REPORT_METRICS
    lines = []
    if fmt == "markdown":
        header = "| Regime | Model | " + " | ".join(m.upper() for m in metrics) + " |"
        sep = "|" + "|".join(["---"] * (len(metrics) + 2)) + "|"
        lines += [header, sep]
        for regime in REGIMES:
            for mode in MODES:
                cells = []
                for metric in metrics:
                    agg = results[regime][mode].get(metric)
                    cells.append(f"{agg['mean']:.3f} ± {agg['std']:.3f}" if agg else "-")
                lines.append(f"| {regime} | {MODE_DISPLAY[mode]} | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    else:  # latex
        col = "ll" + "c" * len(metrics)
        lines.append("\\begin{tabular}{" + col + "}")
        lines.append("\\toprule")
        lines.append("Regime & Model & " + " & ".join(m.upper() for m in metrics) + " \\\\")
        lines.append("\\midrule")
        for regime in REGIMES:
            for mode in MODES:
                cells = []
                for metric in metrics:
                    agg = results[regime][mode].get(metric)
                    cells.append(f"{agg['mean']:.3f} $\\pm$ {agg['std']:.3f}" if agg else "-")
                lines.append(f"{regime} & {MODE_DISPLAY[mode]} & " + " & ".join(cells) + " \\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        return "\n".join(lines)


def make_figure(out: dict, save_path: Path) -> None:
    results = out["results"]
    modes = list(MODES)
    x = np.arange(len(modes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"transductive": "#4C72B0", "inductive": "#C44E52"}
    for i, regime in enumerate(REGIMES):
        means = [results[regime][m]["auroc"]["mean"] for m in modes]
        stds = [results[regime][m]["auroc"]["std"] for m in modes]
        ax.bar(
            x + (i - 0.5) * width, means, width, yerr=stds, capsize=4,
            label=regime, color=colors.get(regime), alpha=0.9,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_DISPLAY[m] for m in modes], rotation=15, ha="right")
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Transductive vs inductive link prediction (PrimeKG-Celiac)")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast 1-seed CPU check with fallback features")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--holdout-side", choices=["src", "dst"], default="src")
    parser.add_argument("--fallback-features", action="store_true", help="Use hashing features (skip transformer)")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "inductive_primekg.json"))
    args = parser.parse_args()

    if args.smoke:
        seeds = args.seeds or [0]
        epochs = 30
        fallback = True
        out_path = Path(args.out).with_name("inductive_primekg_smoke.json")
    else:
        seeds = args.seeds or DEFAULT_SEEDS
        epochs = args.epochs
        fallback = args.fallback_features
        out_path = Path(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seeds: {seeds} | epochs: {epochs} | features: {'hash' if fallback else 'transformer'}")

    out = run(
        seeds=seeds, epochs=epochs, patience=args.patience, lr=args.lr,
        hidden=args.hidden, num_layers=args.layers, dropout=args.dropout,
        holdout_side=args.holdout_side, fallback_features=fallback, device=device,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")

    md = make_table(out, "markdown")
    latex = make_table(out, "latex")
    (out_path.with_suffix(".md")).write_text(md + "\n")
    (out_path.with_name(out_path.stem + "_table.tex")).write_text(latex + "\n")
    print("\n" + md)

    fig_name = "inductive_vs_transductive_smoke.png" if args.smoke else "inductive_vs_transductive.png"
    make_figure(out, FIGURES_DIR / fig_name)

    print("\nPaired tests (feature_gnn > other), AUROC:")
    for regime in REGIMES:
        for comp, st in out["stats"][regime].items():
            print(f"  [{regime}] {comp}: Δ={st['mean_diff']:+.3f} p={st['p_value']:.4f} d={st['cohens_d']:+.2f}")


if __name__ == "__main__":
    main()
