#!/usr/bin/env python
"""Statistical hardening of OncoEvidence headline numbers (deliverable 3).

Two additions make the headline claims defensible rather than single-point:

A) BOOTSTRAP 95% CIs (percentile, paired resampling of the scored pairs) for
   - the mechanism-separation AUROC (true oncology indications vs random pairs,
     point ~0.879; reproduces scripts/evaluate_mechanism.py), and
   - each hard-negative AUROC (random ~0.887, oncology-drug ~0.870,
     degree-matched ~0.742, shared-target ~0.609; reproduces
     scripts/evaluate_hard_negatives.py).
   Resampling positives and negatives independently with replacement and
   recomputing AUROC each draw gives a sampling-variability interval, so a
   reader can see which separations are robust and which (shared_target) are
   close to chance.

B) MULTI-CUTOFF prospective temporal split at T in {2000, 2005, 2010}, training
   the graph HeteroGNN and the structure-blind FeatureMLP at each cutoff and
   reporting GNN-vs-MLP AUROC + the trend across cutoffs. This shows the
   prospective claim is not an artifact of the single T=2005 cutoff. Years are
   reused from the Europe PMC cache (data/temporal_year_cache.json); NO network.

Run:
    PYTHONPATH=. python scripts/stats_hardening.py
    PYTHONPATH=. python scripts/stats_hardening.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))
sys.path.insert(0, _THIS)  # import sibling scripts as modules

import numpy as np
import torch
from scipy.stats import rankdata

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.evaluation.metrics import compute_all_metrics
from oncorepurpose.evaluation.temporal_split import (
    oncology_disease_set, temporal_split, true_oncology_pairs,
)
from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn, train_mlp
from oncorepurpose.interpret.mechanism_paths import build_mech_index
from oncorepurpose.interpret.paths import _known_pairs
from oncorepurpose.models import FeatureMLP, HeteroGNN

# Reuse the EXACT sampling / scoring logic from the existing evaluations so the
# bootstrap is around the same point estimates the README reports.
import evaluate_mechanism as EM
import evaluate_hard_negatives as HN
from evaluate_temporal_split import CACHE_PATH, _query_token

HIDDEN = 128
CUTOFFS = (2000, 2005, 2010)


# --------------------------------------------------------------------------- #
# Bootstrap AUROC
# --------------------------------------------------------------------------- #
def auroc_fast(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-aware AUROC via the Mann-Whitney U / rank-sum identity."""
    n1, n0 = pos.size, neg.size
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int, seed: int = 0):
    """Percentile 95% CI for AUROC, resampling pos/neg independently w/ replacement."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    point = auroc_fast(pos, neg)
    rng = np.random.default_rng(seed)
    n1, n0 = pos.size, neg.size
    draws = np.empty(n_boot, float)
    for b in range(n_boot):
        bp = pos[rng.integers(0, n1, n1)]
        bn = neg[rng.integers(0, n0, n0)]
        draws[b] = auroc_fast(bp, bn)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"auroc": point, "ci95_low": float(lo), "ci95_high": float(hi),
            "boot_mean": float(draws.mean()), "boot_std": float(draws.std()),
            "n_pos": int(n1), "n_neg": int(n0), "n_boot": int(n_boot)}


# --------------------------------------------------------------------------- #
# Part A: mechanism-separation + hard-negative bootstrap CIs
# --------------------------------------------------------------------------- #
def mechanism_separation_scores(data, idx):
    """Reproduce scripts/evaluate_mechanism.py (point ~0.879): true vs random."""
    true_pairs, neg_pairs = EM.sample_pairs(data, idx)
    s_true, _ = EM.score_group(data, idx, true_pairs)
    s_neg, _ = EM.score_group(data, idx, neg_pairs)
    return s_true, s_neg


def hard_negative_scores(data, idx):
    """Reproduce scripts/evaluate_hard_negatives.py true-pair + negative-set scores."""
    onco_set = HN.oncology_disease_indices(data)
    onco_list = sorted(onco_set)
    known = _known_pairs(data)
    num_drugs = int(data[DRUG_TYPE].num_nodes)
    num_dis = int(data[DISEASE_TYPE].num_nodes)

    ind_et = (DRUG_TYPE, "indication", DISEASE_TYPE)
    ei = data[ind_et].edge_index
    if ind_et[0] == DRUG_TYPE:
        ind_drug, ind_dis = ei[0].tolist(), ei[1].tolist()
    else:
        ind_drug, ind_dis = ei[1].tolist(), ei[0].tolist()

    rng = random.Random(HN.SEED)
    true_pairs = [(dr, ds) for dr, ds in zip(ind_drug, ind_dis) if ds in onco_set]
    rng.shuffle(true_pairs)
    true_pairs = true_pairs[: HN.N_TRUE]
    onco_drugs = {dr for dr, ds in zip(ind_drug, ind_dis) if ds in onco_set}

    drug_deg = HN.node_degrees(data, DRUG_TYPE, num_drugs)
    dis_deg = HN.node_degrees(data, DISEASE_TYPE, num_dis)
    drug_bin, drug_bucket = HN.decile_bins(drug_deg, range(num_drugs))
    dis_bin, dis_bucket = HN.decile_bins(dis_deg, onco_set)

    s_true = HN.pair_scores(data, idx, true_pairs)[0]

    negsets = {
        "random": HN.sample_random(random.Random(HN.SEED + 1), true_pairs, known,
                                   num_drugs, onco_list),
        "degree_matched": HN.sample_degree_matched(random.Random(HN.SEED + 2), true_pairs,
                                                   known, drug_bin, drug_bucket,
                                                   dis_bin, dis_bucket),
        "oncology_drug": HN.sample_oncology_drug(random.Random(HN.SEED + 3), true_pairs,
                                                known, onco_drugs, onco_list),
    }
    try:
        st = HN.sample_shared_target(random.Random(HN.SEED + 4), true_pairs, known, idx, onco_set)
        if st:
            negsets["shared_target"] = st
    except Exception as exc:  # noqa: BLE001
        print(f"shared_target skipped: {exc}")

    neg_scores = {name: HN.pair_scores(data, idx, negs)[0] for name, negs in negsets.items()}
    return s_true, neg_scores


# --------------------------------------------------------------------------- #
# Part B: multi-cutoff temporal split (cached years, no network)
# --------------------------------------------------------------------------- #
def pair_years_from_cache(data, target, n_pairs, sample_seed=0):
    """Build {(drug_idx, dis_idx): year} from the Europe PMC cache only."""
    if not CACHE_PATH.exists():
        raise SystemExit(f"temporal year cache not found at {CACHE_PATH}")
    cache = json.loads(CACHE_PATH.read_text())
    drug_names = list(data[DRUG_TYPE].node_names)
    dis_names = list(data[DISEASE_TYPE].node_names)
    all_pairs = true_oncology_pairs(data, target)
    rng = random.Random(sample_seed)
    sampled = all_pairs[:]
    rng.shuffle(sampled)
    sampled = sampled[:n_pairs]
    pair_years, miss = {}, 0
    for dr, ds in sampled:
        key = f"{_query_token(drug_names[dr]).lower()}||{_query_token(dis_names[ds]).lower()}"
        val = cache.get(key)
        if isinstance(val, int):
            pair_years[(dr, ds)] = val
        else:
            miss += 1
    return pair_years, len(sampled), miss


def train_eval_temporal(data, target, pair_years, cutoff, onco_set, in_dims,
                        device, epochs, mlp_epochs, patience, seeds):
    gnn_au, mlp_au = [], []
    info = None
    for seed in seeds:
        split = temporal_split(data, target, pair_years, cutoff, onco_set=onco_set,
                               neg_ratio=1.0, test_neg_ratio=5.0, val_frac=0.15, seed=seed)
        info = split.info

        set_all_seeds(seed)
        gnn = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims,
                        hidden=HIDDEN, num_layers=2, dropout=0.3)
        gnn = train_gnn(gnn, split, device, epochs=epochs, patience=patience)
        with torch.no_grad():
            gnn.eval()
            z = gnn.encode(split.base)
            sc = torch.sigmoid(gnn.decode(z, target, split.test_label_index)).cpu().numpy()
        gnn_au.append(compute_all_metrics(split.test_label.cpu().numpy(), sc)["auroc"])

        set_all_seeds(seed)
        mlp = FeatureMLP(list(data.node_types), in_dims, hidden=HIDDEN, dropout=0.3)
        mlp = train_mlp(mlp, split, device, epochs=mlp_epochs, patience=patience)
        with torch.no_grad():
            mlp.eval()
            z = mlp.encode(split.base)
            sc = torch.sigmoid(mlp.decode(z, target, split.test_label_index)).cpu().numpy()
        mlp_au.append(compute_all_metrics(split.test_label.cpu().numpy(), sc)["auroc"])

    gnn_au = np.asarray(gnn_au)
    mlp_au = np.asarray(mlp_au)
    return {
        "cutoff_year": int(cutoff),
        "n_past_total": info["n_past_total"],
        "n_train_pos": info["n_train_pos"],
        "n_future_pos": info["n_future_pos"],
        "gnn_auroc_mean": float(gnn_au.mean()), "gnn_auroc_std": float(gnn_au.std()),
        "mlp_auroc_mean": float(mlp_au.mean()), "mlp_auroc_std": float(mlp_au.std()),
        "graph_gain_auroc": float(gnn_au.mean() - mlp_au.mean()),
        "gnn_auroc_values": [float(x) for x in gnn_au],
        "mlp_auroc_values": [float(x) for x in mlp_au],
    }


def _device_with_fallback_run(fn, *args, **kwargs):
    """Run `fn(..., device=...)` on CUDA, retrying on CPU if CUDA OOMs."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return fn(*args, device=dev, **kwargs), dev
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and dev.type == "cuda":
            print("  [oom] CUDA OOM in temporal training -> CPU")
            torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            return fn(*args, device=cpu, **kwargs), cpu
        raise


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--mlp-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-pairs", type=int, default=350)
    ap.add_argument("--cutoffs", type=int, nargs="+", default=list(CUTOFFS))
    args = ap.parse_args()

    n_boot = args.n_boot or (300 if args.smoke else 2000)
    epochs = args.epochs or (12 if args.smoke else 40)
    mlp_epochs = args.mlp_epochs or (40 if args.smoke else 150)
    seeds = args.seeds or ([0] if args.smoke else [0, 1])

    print(f"device(cuda)={torch.cuda.is_available()} | n_boot={n_boot} "
          f"epochs={epochs} mlp_epochs={mlp_epochs} seeds={seeds} cutoffs={args.cutoffs}")

    data, targets = load_primekg(with_features=True)
    target = targets["indication"]
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}
    onco_set = oncology_disease_set(data)

    print("building mechanism index ...")
    idx = build_mech_index(data)

    # ---- Part A: bootstrap CIs ------------------------------------------- #
    print("\n[A] mechanism-separation bootstrap (true vs random) ...")
    s_true_sep, s_neg_sep = mechanism_separation_scores(data, idx)
    sep_ci = bootstrap_ci(s_true_sep, s_neg_sep, n_boot, seed=0)
    print(f"  separation AUROC={sep_ci['auroc']:.3f} "
          f"95% CI [{sep_ci['ci95_low']:.3f}, {sep_ci['ci95_high']:.3f}]")

    print("[A] hard-negative bootstrap CIs ...")
    s_true_hn, neg_scores = hard_negative_scores(data, idx)
    hard_neg_ci = {}
    for name, sneg in neg_scores.items():
        ci = bootstrap_ci(s_true_hn, sneg, n_boot, seed=0)
        hard_neg_ci[name] = ci
        print(f"  {name:<15} AUROC={ci['auroc']:.3f} "
              f"95% CI [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}] (n_neg={ci['n_neg']})")

    # ---- Part B: multi-cutoff temporal ----------------------------------- #
    print("\n[B] multi-cutoff prospective temporal split (cached years) ...")
    pair_years, n_sampled, n_miss = pair_years_from_cache(data, target, args.n_pairs)
    print(f"  pairs sampled={n_sampled}, years from cache={len(pair_years)}, "
          f"unresolved(skipped)={n_miss}")
    if len(pair_years) < 10:
        raise SystemExit("too few cached years to run temporal split")

    temporal = []
    for T in args.cutoffs:
        n_future = sum(1 for y in pair_years.values() if y > T)
        n_past = sum(1 for y in pair_years.values() if y <= T)
        if n_future < 5 or n_past < 5:
            print(f"  cutoff T={T}: skipped (past={n_past}, future={n_future})")
            continue
        row, dev_used = _device_with_fallback_run(
            train_eval_temporal, data, target, pair_years, T, onco_set, in_dims,
            epochs=epochs, mlp_epochs=mlp_epochs, patience=args.patience, seeds=seeds)
        temporal.append(row)
        print(f"  T={T}: PAST={row['n_train_pos']}(+val) FUTURE={row['n_future_pos']} | "
              f"GNN AUROC={row['gnn_auroc_mean']:.3f}±{row['gnn_auroc_std']:.3f} | "
              f"MLP AUROC={row['mlp_auroc_mean']:.3f}±{row['mlp_auroc_std']:.3f} | "
              f"gain {row['graph_gain_auroc']:+.3f}")

    # Trend across cutoffs (slope of GNN AUROC and of graph gain vs cutoff year).
    trend = None
    if len(temporal) >= 2:
        xs = np.array([r["cutoff_year"] for r in temporal], float)
        gnn_y = np.array([r["gnn_auroc_mean"] for r in temporal], float)
        gain_y = np.array([r["graph_gain_auroc"] for r in temporal], float)
        gnn_slope = float(np.polyfit(xs, gnn_y, 1)[0])
        gain_slope = float(np.polyfit(xs, gain_y, 1)[0])
        trend = {
            "cutoffs": [int(x) for x in xs],
            "gnn_auroc_per_cutoff": [float(y) for y in gnn_y],
            "graph_gain_per_cutoff": [float(y) for y in gain_y],
            "gnn_auroc_slope_per_year": gnn_slope,
            "graph_gain_slope_per_year": gain_slope,
            "gnn_above_chance_all": bool(np.all(gnn_y > 0.5)),
            "graph_gain_positive_all": bool(np.all(gain_y > 0.0)),
        }

    # ---- Assemble + persist ---------------------------------------------- #
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "dataset": "PrimeKG",
        "n_boot": n_boot,
        "bootstrap": {
            "mechanism_separation": sep_ci,
            "hard_negatives": hard_neg_ci,
        },
        "temporal_multi_cutoff": {
            "config": {"n_pairs": args.n_pairs, "epochs": epochs,
                       "mlp_epochs": mlp_epochs, "seeds": seeds,
                       "years_from_cache": len(pair_years), "unresolved": n_miss},
            "per_cutoff": temporal,
            "trend": trend,
        },
    }
    headline = build_headline(result)
    result["headline"] = headline

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "stats_hardening.json"
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(RESULTS_DIR / "stats_hardening.md", result)
    print("\n" + headline)
    print(f"saved -> {out_json}")
    print(f"saved -> {RESULTS_DIR / 'stats_hardening.md'}")


def build_headline(r):
    sep = r["bootstrap"]["mechanism_separation"]
    parts = [
        f"Mechanism separation AUROC {sep['auroc']:.3f} "
        f"(95% CI {sep['ci95_low']:.3f}-{sep['ci95_high']:.3f})."
    ]
    t = r["temporal_multi_cutoff"]["trend"]
    if t:
        gains = ", ".join(f"T={c}:{g:+.3f}" for c, g in
                          zip(t["cutoffs"], t["graph_gain_per_cutoff"]))
        parts.append(
            f"Prospective GNN AUROC stays above chance at all cutoffs "
            f"({'yes' if t['gnn_above_chance_all'] else 'no'}); graph gain over MLP "
            f"[{gains}]."
        )
    return " ".join(parts)


def write_markdown(path, r):
    sep = r["bootstrap"]["mechanism_separation"]
    lines = [
        "# OncoEvidence: statistical hardening",
        "",
        f"_{r['timestamp_utc']}_  ·  mode: {r['mode']}  ·  bootstrap draws: {r['n_boot']}",
        "",
        "## Headline",
        "",
        r["headline"],
        "",
        "## A. Bootstrap 95% CIs (AUROC)",
        "",
        "Percentile CIs from resampling the scored pairs (positives and negatives "
        "independently, with replacement).",
        "",
        "| Comparison | AUROC | 95% CI | n_pos | n_neg |",
        "|---|---|---|---|---|",
        f"| Mechanism separation (true vs random) | {sep['auroc']:.3f} | "
        f"[{sep['ci95_low']:.3f}, {sep['ci95_high']:.3f}] | {sep['n_pos']} | {sep['n_neg']} |",
    ]
    for name, ci in r["bootstrap"]["hard_negatives"].items():
        lines.append(
            f"| Hard-negative: {name} | {ci['auroc']:.3f} | "
            f"[{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}] | {ci['n_pos']} | {ci['n_neg']} |"
        )
    lines += [
        "",
        "## B. Multi-cutoff prospective temporal split",
        "",
        "Earliest-evidence years reused from the Europe PMC cache (no network). "
        "FUTURE indication edges (year > T) are held out of the message-passing "
        "graph; the model must rank them above sampled negatives using only PAST "
        "structure. GNN (graph) vs FeatureMLP (structure-blind).",
        "",
        "| Cutoff T | PAST train + | FUTURE test + | GNN AUROC | MLP AUROC | Graph gain |",
        "|---|---|---|---|---|---|",
    ]
    for row in r["temporal_multi_cutoff"]["per_cutoff"]:
        lines.append(
            f"| {row['cutoff_year']} | {row['n_train_pos']} | {row['n_future_pos']} | "
            f"{row['gnn_auroc_mean']:.3f} ± {row['gnn_auroc_std']:.3f} | "
            f"{row['mlp_auroc_mean']:.3f} ± {row['mlp_auroc_std']:.3f} | "
            f"{row['graph_gain_auroc']:+.3f} |"
        )
    t = r["temporal_multi_cutoff"]["trend"]
    if t:
        lines += [
            "",
            f"- GNN AUROC trend across cutoffs: slope **{t['gnn_auroc_slope_per_year']:+.4f}/yr**; "
            f"above chance at every cutoff: **{t['gnn_above_chance_all']}**.",
            f"- Graph-gain trend (GNN−MLP): slope **{t['graph_gain_slope_per_year']:+.4f}/yr**; "
            f"positive at every cutoff: **{t['graph_gain_positive_all']}**.",
        ]
    lines += [
        "",
        "## Honest read & caveats",
        "",
        "- A bootstrap CI captures sampling variability of the *scored pairs*, not "
        "the upstream choices (pair sampling, negative construction); the "
        "shared_target CI sitting near 0.5–0.6 confirms the mechanism signal is "
        "weak against same-target decoys, exactly as the point estimate warned.",
        "- The temporal axis is an *approximate* first-evidence proxy (earliest "
        "Europe PMC co-mention), so absolute AUROCs are indicative; the value here "
        "is the consistency of the GNN-over-MLP gap across multiple cutoffs.",
        "- Fewer FUTURE positives at later cutoffs widen the effective uncertainty; "
        "trends across only three cutoffs are directional, not a fitted law.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
