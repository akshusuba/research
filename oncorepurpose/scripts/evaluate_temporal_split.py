#!/usr/bin/env python
"""Temporal-split PROSPECTIVE evaluation for OncoEvidence.

Question: is the pipeline genuinely *predictive*? Could it have ranked a true
drug->cancer indication ABOVE random (drug, cancer) pairs using only structure
that existed BEFORE that indication was established -- rather than being merely
retrospectively consistent?

PrimeKG has no edge timestamps, so we derive an approximate first-evidence YEAR
per true oncology indication pair from Europe PMC (earliest publication that
co-mentions the drug AND the disease). With a cutoff year T:
  - PAST pairs (year <= T) seed the message-passing graph + train supervision.
  - FUTURE pairs (year > T) are held out; their edges are removed from the graph
    (no leakage) and become the prospective test positives.
We then ask whether the HeteroGNN (graph) and a structure-blind FeatureMLP
control can rank FUTURE true indications above sampled negatives, and -- the
interesting part -- whether the GRAPH beats the structure-blind control.

Run:
    PYTHONPATH=. python scripts/evaluate_temporal_split.py --smoke   # fast sanity
    PYTHONPATH=. python scripts/evaluate_temporal_split.py           # full eval
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests
import torch

from oncorepurpose.config import DATA_DIR, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.evaluation.metrics import compute_all_metrics
from oncorepurpose.evaluation.temporal_split import (
    oncology_disease_set,
    temporal_split,
    true_oncology_pairs,
)
from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn, train_mlp
from oncorepurpose.models import FeatureMLP, HeteroGNN

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CACHE_PATH = DATA_DIR / "temporal_year_cache.json"
CURRENT_YEAR = datetime.now().year
HIDDEN = 128


# --------------------------------------------------------------------------- #
# Europe PMC first-evidence year lookup (cached, polite, failure-tolerant)
# --------------------------------------------------------------------------- #
def _query_token(name: str) -> str:
    """Drop trailing parenthetical qualifiers; collapse whitespace; strip quotes."""
    base = re.split(r"\s*\(", name or "")[0]
    return re.sub(r"\s+", " ", base.replace('"', " ")).strip()


def earliest_year(session, drug: str, disease: str, sleep: float):
    """Earliest Europe PMC co-mention year for (drug, disease).

    Returns (status, year): status in {"ok", "empty", "error"}; year is int|None.
    "ok"/"empty" are cacheable (definitive); "error" (network/HTTP) is not.
    """
    d, c = _query_token(drug), _query_token(disease)
    if not d or not c:
        return "empty", None
    # Unquoted (implicit-AND) co-mention query. Exact-phrase quoting was found to
    # surface spurious "earliest" records (e.g. a 2026-dated hit for methotrexate +
    # non-Hodgkin lymphoma, whose true first co-mention is 1960), so we follow the
    # plain `<drug> AND <disease>` form which dates first evidence far more reliably.
    query = f"{d} AND {c}"
    try:
        # FIRST_PDATE_D asc (not P_PDATE_D asc): the latter sorts records with a
        # missing print-publication date (PMC entries, preprints) FIRST, which
        # surfaces spurious current-year "earliest" hits (e.g. imatinib + GIST
        # dated 2026 instead of its true 2001 first evidence). FIRST_PDATE_D
        # sorts on first publication date and dates first evidence correctly.
        r = session.get(
            EUROPE_PMC,
            params={"query": query, "sort": "FIRST_PDATE_D asc", "format": "json",
                    "pageSize": 1, "resultType": "lite"},
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("resultList", {}).get("result", [])
    except Exception as exc:
        print(f"  [epmc] error '{d}'+'{c}': {exc}")
        return "error", None
    finally:
        time.sleep(sleep)
    if not results:
        return "empty", None
    raw = results[0].get("firstPublicationDate", "")[:4] or results[0].get("pubYear")
    try:
        y = int(raw)
    except (TypeError, ValueError):
        return "empty", None
    if y < 1900 or y > CURRENT_YEAR:
        return "empty", None
    return "ok", y


def collect_years(pairs, drug_names, dis_names, sleep, verbose=True):
    """Resolve {(drug_idx, dis_idx): year} for `pairs`, using/refreshing cache."""
    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            cache = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "OncoEvidence-temporal-eval/1.0"})

    pair_years, n_query, n_cache, n_empty = {}, 0, 0, 0
    for i, (dr, ds) in enumerate(pairs):
        drug, disease = drug_names[dr], dis_names[ds]
        key = f"{_query_token(drug).lower()}||{_query_token(disease).lower()}"
        if key in cache:
            n_cache += 1
            val = cache[key]
            if isinstance(val, int):
                pair_years[(dr, ds)] = val
            else:
                n_empty += 1
            continue
        status, year = earliest_year(session, drug, disease, sleep)
        n_query += 1
        if status == "ok":
            cache[key] = year
            pair_years[(dr, ds)] = year
        elif status == "empty":
            cache[key] = None
            n_empty += 1
        # "error" -> do not cache (retry on a later run)
        if verbose and (i + 1) % 25 == 0:
            print(f"  resolved {i+1}/{len(pairs)} pairs "
                  f"(cache {n_cache}, new {n_query}, no-year {n_empty})")
        if (i + 1) % 50 == 0:
            CACHE_PATH.write_text(json.dumps(cache, indent=0))
    CACHE_PATH.write_text(json.dumps(cache, indent=0))
    print(f"years resolved: {len(pair_years)}/{len(pairs)} "
          f"(from cache: {n_cache}, queried: {n_query}, no-year: {n_empty})")
    return pair_years


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_scores)
    return float(y_true[order[:k]].sum() / y_true.sum())


@torch.no_grad()
def _scores(model, split, device) -> np.ndarray:
    model.eval()
    z = model.encode(split.base)
    return torch.sigmoid(model.decode(z, split.target_edge_type, split.test_label_index)).cpu().numpy()


def eval_split(model, split, device, ks) -> dict:
    scores = _scores(model, split, device)
    y = split.test_label.cpu().numpy()
    out = compute_all_metrics(y, scores)
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(y, scores, min(k, len(y)))
    return out


def train_eval_models(split, data, in_dims, device, epochs, mlp_epochs, patience, seed, ks):
    results = {}
    set_all_seeds(seed)
    gnn = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims,
                    hidden=HIDDEN, num_layers=2, dropout=0.3)
    gnn = train_gnn(gnn, split, device, epochs=epochs, patience=patience)
    results["GNN"] = eval_split(gnn, split, device, ks)

    set_all_seeds(seed)
    mlp = FeatureMLP(list(data.node_types), in_dims, hidden=HIDDEN, dropout=0.3)
    mlp = train_mlp(mlp, split, device, epochs=mlp_epochs, patience=patience)
    results["MLP"] = eval_split(mlp, split, device, ks)
    return results


def aggregate(per_seed_metrics):
    """per_seed_metrics: list of dicts -> {metric: {mean, std, values}}."""
    keys = set().union(*[m.keys() for m in per_seed_metrics]) if per_seed_metrics else set()
    out = {}
    for k in sorted(keys):
        vals = [float(m[k]) for m in per_seed_metrics if k in m]
        a = np.asarray(vals, float)
        out[k] = {"mean": float(a.mean()), "std": float(a.std()), "values": vals}
    return out


def choose_cutoff(years, percentile):
    """Pick T so that year<=T is PAST; ensure both PAST and FUTURE are non-empty."""
    arr = np.sort(np.asarray(years, dtype=int))
    T = int(np.percentile(arr, percentile))
    # Guarantee at least one FUTURE pair.
    while T >= int(arr.min()) and (arr > T).sum() == 0:
        T -= 1
    # Guarantee at least one PAST pair.
    while T < int(arr.max()) and (arr <= T).sum() == 0:
        T += 1
    return T


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end sanity run (few pairs, few epochs, 1 seed)")
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="number of true oncology pairs to sample for year lookup")
    ap.add_argument("--epochs", type=int, default=None, help="GNN epochs")
    ap.add_argument("--mlp-epochs", type=int, default=None, help="MLP epochs")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--percentile", type=float, default=70.0,
                    help="cutoff-year percentile (PAST fraction)")
    ap.add_argument("--sleep", type=float, default=0.2, help="polite delay between EPMC calls")
    ap.add_argument("--sample-seed", type=int, default=0, help="seed for pair sampling")
    args = ap.parse_args()

    if args.smoke:
        n_pairs = args.n_pairs or 16
        epochs = args.epochs or 12
        mlp_epochs = args.mlp_epochs or 40
        seeds = args.seeds or [0]
        ks = [5, 10]
        out_json = RESULTS_DIR / "temporal_split_eval_smoke.json"
        out_md = RESULTS_DIR / "temporal_split_eval_smoke.md"
    else:
        n_pairs = args.n_pairs or 350
        epochs = args.epochs or 50
        mlp_epochs = args.mlp_epochs or 200
        seeds = args.seeds or [0, 1, 2]
        ks = [50, 100, 200]
        out_json = RESULTS_DIR / "temporal_split_eval.json"
        out_md = RESULTS_DIR / "temporal_split_eval.md"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | mode={'smoke' if args.smoke else 'full'} | "
          f"n_pairs={n_pairs} epochs={epochs} seeds={seeds}")

    data, targets = load_primekg(with_features=True)
    target = targets["indication"]
    print(f"target edge type: {target}")
    drug_names = list(data[DRUG_TYPE].node_names)
    dis_names = list(data["disease"].node_names)
    onco_set = oncology_disease_set(data)
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}

    all_pairs = true_oncology_pairs(data, target)
    print(f"true oncology indication pairs: {len(all_pairs)} | oncology diseases: {len(onco_set)}")
    rng = random.Random(args.sample_seed)
    sampled = all_pairs[:]
    rng.shuffle(sampled)
    sampled = sampled[:n_pairs]

    pair_years = collect_years(sampled, drug_names, dis_names, args.sleep)
    if len(pair_years) < 4:
        raise SystemExit(f"Too few resolved years ({len(pair_years)}); cannot form a split.")

    years = list(pair_years.values())
    cutoff = choose_cutoff(years, args.percentile)
    n_past = sum(1 for y in years if y <= cutoff)
    n_future = sum(1 for y in years if y > cutoff)
    yr_arr = np.asarray(years)
    print(f"\nyear distribution: min={yr_arr.min()} p25={np.percentile(yr_arr,25):.0f} "
          f"median={np.median(yr_arr):.0f} p75={np.percentile(yr_arr,75):.0f} max={yr_arr.max()}")
    print(f"cutoff T={cutoff} (p{args.percentile:.0f}) -> PAST={n_past}, FUTURE={n_future}")

    per_seed = {"GNN": [], "MLP": []}
    split_info = None
    for seed in seeds:
        split = temporal_split(data, target, pair_years, cutoff, onco_set=onco_set,
                               neg_ratio=1.0, test_neg_ratio=(2.0 if args.smoke else 5.0),
                               val_frac=0.15, seed=seed)
        split_info = split.info
        res = train_eval_models(split, data, in_dims, device, epochs, mlp_epochs,
                                args.patience, seed, ks)
        for m in ("GNN", "MLP"):
            per_seed[m].append(res[m])
        print(f"  seed {seed}: "
              f"GNN auroc={res['GNN']['auroc']:.3f} auprc={res['GNN']['auprc']:.3f} | "
              f"MLP auroc={res['MLP']['auroc']:.3f} auprc={res['MLP']['auprc']:.3f}")

    agg = {m: aggregate(per_seed[m]) for m in ("GNN", "MLP")}

    gnn_auroc = [d["auroc"] for d in per_seed["GNN"]]
    mlp_auroc = [d["auroc"] for d in per_seed["MLP"]]
    gnn_mean = float(np.mean(gnn_auroc))
    mlp_mean = float(np.mean(mlp_auroc))
    graph_gain = gnn_mean - mlp_mean
    above_chance = gnn_mean > 0.5

    headline = (
        f"Prospective (temporal) test on FUTURE oncology indications (cutoff T={cutoff}): "
        f"GNN AUROC={gnn_mean:.3f}, structure-blind MLP AUROC={mlp_mean:.3f} "
        f"(graph gain {graph_gain:+.3f})."
    )

    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "dataset": "PrimeKG",
        "target_edge_type": list(target),
        "config": {
            "n_pairs_sampled": n_pairs, "epochs": epochs, "mlp_epochs": mlp_epochs,
            "patience": args.patience, "seeds": seeds, "percentile": args.percentile,
            "sample_seed": args.sample_seed, "recall_ks": ks, "hidden": HIDDEN,
            "test_neg_ratio": (2.0 if args.smoke else 5.0),
        },
        "cutoff_year": cutoff,
        "counts": {
            "true_oncology_pairs_total": len(all_pairs),
            "sampled": n_pairs,
            "years_resolved": len(pair_years),
            "past": n_past,
            "future": n_future,
        },
        "year_distribution": {
            "min": int(yr_arr.min()), "p25": float(np.percentile(yr_arr, 25)),
            "median": float(np.median(yr_arr)), "p75": float(np.percentile(yr_arr, 75)),
            "max": int(yr_arr.max()),
        },
        "split_info": split_info,
        "results": agg,
        "per_seed": per_seed,
        "comparison": {
            "gnn_auroc_mean": gnn_mean, "mlp_auroc_mean": mlp_mean,
            "graph_gain_auroc": graph_gain, "gnn_above_chance": above_chance,
        },
        "headline": headline,
        "device": str(device),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(out_md, result)
    print(f"\n{headline}")
    print(f"saved -> {out_json}")
    print(f"saved -> {out_md}")


def _fmt(stat):
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def write_markdown(path, r):
    c = r["counts"]
    g, m = r["results"]["GNN"], r["results"]["MLP"]
    comp = r["comparison"]
    ks = r["config"]["recall_ks"]
    verdict_chance = ("**above chance**" if comp["gnn_above_chance"]
                      else "**not above chance**")
    if comp["graph_gain_auroc"] > 0.01:
        verdict_graph = ("the **graph adds value**: the GNN beats the structure-blind "
                         f"control by {comp['graph_gain_auroc']:+.3f} AUROC on this "
                         "prospective task.")
    elif comp["graph_gain_auroc"] < -0.01:
        verdict_graph = ("the **graph does NOT add value here**: the structure-blind "
                         f"control matches or beats the GNN ({comp['graph_gain_auroc']:+.3f} "
                         "AUROC).")
    else:
        verdict_graph = ("the **graph and the structure-blind control are roughly tied** "
                         f"({comp['graph_gain_auroc']:+.3f} AUROC) on this prospective task.")

    lines = [
        f"# Temporal-split prospective evaluation ({r['mode']})",
        "",
        f"_{r['timestamp']}_",
        "",
        "## Headline",
        "",
        r["headline"],
        "",
        f"The GNN ranks FUTURE true oncology indications {verdict_chance} above random "
        f"(drug, cancer) negatives, and {verdict_graph}",
        "",
        "## Setup",
        "",
        f"- Temporal axis: earliest Europe PMC year co-mentioning each true "
        f"(drug, cancer) indication pair.",
        f"- Cutoff **T = {r['cutoff_year']}** (p{r['config']['percentile']:.0f} of "
        f"resolved years).",
        f"- True oncology indication pairs total: {c['true_oncology_pairs_total']}; "
        f"sampled {c['sampled']}; years resolved {c['years_resolved']}.",
        f"- **PAST** (year ≤ T, in message-passing graph + train): {c['past']}; "
        f"**FUTURE** (year > T, held-out prospective test): {c['future']}.",
        f"- FUTURE target edges removed from the graph (no leakage); all drug↔disease "
        f"therapeutic edges stripped exactly as in the inductive split.",
        f"- Test negatives: random (drug, oncology-cancer) pairs at "
        f"{r['config']['test_neg_ratio']}× positives; {len(r['config']['seeds'])} seed(s).",
        "",
        "## Results (mean ± std over seeds)",
        "",
        "| Model | AUROC | AUPRC | " + " | ".join(f"recall@{k}" for k in ks) + " |",
        "|---|---|---|" + "---|" * len(ks),
        f"| **GNN (graph)** | {_fmt(g['auroc'])} | {_fmt(g['auprc'])} | "
        + " | ".join(_fmt(g[f'recall@{k}']) for k in ks) + " |",
        f"| MLP (structure-blind) | {_fmt(m['auroc'])} | {_fmt(m['auprc'])} | "
        + " | ".join(_fmt(m[f'recall@{k}']) for k in ks) + " |",
        "",
        "## Honest read",
        "",
        f"- Prospective AUROC (GNN) = **{comp['gnn_auroc_mean']:.3f}** vs 0.5 chance.",
        f"- Graph vs structure-blind control: **{comp['graph_gain_auroc']:+.3f}** AUROC.",
        "- Caveats: first-evidence year is an *approximate* proxy (earliest literature "
        "co-mention, not regulatory/discovery date) and is noisy; the sample is a seeded "
        "subset of oncology pairs; Europe PMC co-mention can precede or lag true "
        "establishment of an indication. Treat magnitudes as indicative, not exact.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
