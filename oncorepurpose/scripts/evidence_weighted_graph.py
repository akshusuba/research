#!/usr/bin/env python
"""Evidence-weighted mechanism scoring -- make the retrieval layer LOAD-BEARING.

The structure-only mechanism score (``oncorepurpose.interpret.mechanism_paths``)
ranks a (drug, cancer) pair by graph topology alone: a direct drug->target->cancer
chain scores ~3 whether or not the literature has ever connected that drug to that
target. That lets *coincidental* paths through promiscuous hub genes look just as
strong as a real mechanism. This script re-scores each mechanism path with a
literature-support weight from Europe PMC and asks whether evidence weighting
sharpens the separation of true indications from negatives -- especially the
HARD shared-target negatives, where a coincidental direct-target path is the whole
problem.

Per mechanism path (drug D, cancer C, target gene tg, disease-side gene dg) the
evidence weight combines:
  (a) chain co-mention    : log1p(hits(D, tg)) + log1p(hits(dg, C))  -- both links
      of the mechanism chain must be attested in the literature, not just the graph;
  (b) recency             : exp-decayed most-recent co-mention year of each link;
  (c) specificity         : 1/log10(total_mentions+10) of the bridge gene, so a hub
      gene mentioned with *everything* (TP53, MYC, EGFR, ...) is down-weighted;
  (d) indication evidence : log1p(hits(D, C)) -- is THIS drug actually studied in
      THIS cancer? This is the term that separates true indications from shared-
      target hard negatives, which share the same cancer and a real drug-target edge
      but where the drug is rarely co-mentioned with the cancer;
  (e) contradiction penalty: a pair-level multiplier 1/(1+a*contra) from the lexical
      contradiction scan (``scripts/evidence_lit.py``; cf. contradiction_detector.py).

evidence_weighted_score(pair) = max_path [ struct_score(path) * chain_support(path) ]
                                * log1p(hits(D, C)) / (1 + a * contradiction_count(pair))

We then compare separation AUROC (structure-only vs evidence-weighted) for
true-vs-random AND true-vs-shared-target, and surface concrete hub paths that get
demoted. All Europe PMC calls are cached (``data/europepmc_evidence_cache.json``)
and rate limited (~0.3s between live calls); random negatives mostly have no path,
so they cost almost nothing.

Run:
    PYTHONPATH=. .venv/bin/python scripts/evidence_weighted_graph.py --smoke
    PYTHONPATH=. .venv/bin/python scripts/evidence_weighted_graph.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from oncorepurpose.config import DATA_DIR, DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.interpret.mechanism_paths import (
    build_mech_index, mechanism_paths, mechanism_score,
)
from oncorepurpose.interpret.paths import _known_pairs
from scripts.evidence_lit import (
    EPMCCache, count_recency, recency_weight, contradiction_scan, query_token,
)

CACHE_PATH = DATA_DIR / "europepmc_evidence_cache.json"
SEED = 0
CONTRA_ALPHA = 0.5  # contradiction penalty strength
TOP_K_PATHS = 3     # paths per pair that receive an evidence weight


def oncology_disease_indices(data):
    store = data[DISEASE_TYPE]
    if "is_oncology" in store:
        return set(torch.nonzero(store.is_oncology, as_tuple=False).flatten().tolist())
    return set(range(int(store.num_nodes)))


def sample_pairs(data, idx, n_true, n_neg):
    """True oncology indications, random negatives, and shared-target hard negatives."""
    rng = random.Random(SEED)
    onco = oncology_disease_indices(data)
    onco_list = sorted(onco)
    known = _known_pairs(data)
    ei = data[(DRUG_TYPE, "indication", DISEASE_TYPE)].edge_index
    true_pairs = [(dr, ds) for dr, ds in zip(ei[0].tolist(), ei[1].tolist()) if ds in onco]
    rng.shuffle(true_pairs)
    true_pairs = true_pairs[:n_true]

    num_drugs = int(data[DRUG_TYPE].num_nodes)
    rng_r = random.Random(SEED + 1)
    neg_pairs, seen = [], set()
    while len(neg_pairs) < n_neg and len(seen) < n_neg * 50:
        dr = rng_r.randrange(num_drugs)
        ds = rng_r.choice(onco_list)
        if (dr, ds) in known or (dr, ds) in seen:
            continue
        seen.add((dr, ds))
        neg_pairs.append((dr, ds))

    # Shared-target hard negatives: a different drug sharing >=1 target with the
    # true drug for that SAME cancer but lacking the indication. These trivially get
    # a direct-target path (the shared target IS a disease gene), so they are the
    # acid test for whether evidence weighting demotes coincidental hub paths.
    drug2prot, prot2drug = idx["drug2prot"], idx["prot2drug"]
    rng_s = random.Random(SEED + 4)
    st_pairs, st_seen = [], set()
    for dr_t, ds in true_pairs:
        targets = drug2prot.get(dr_t, set())
        if not targets:
            continue
        cands = set()
        for p in targets:
            cands |= prot2drug.get(p, set())
        cands.discard(dr_t)
        cands = [c for c in cands if (c, ds) not in known and (c, ds) not in st_seen]
        if not cands:
            continue
        dr = rng_s.choice(cands)
        st_seen.add((dr, ds))
        st_pairs.append((dr, ds))
    st_pairs = st_pairs[:n_neg]
    return true_pairs, neg_pairs, st_pairs


def _spec_gene(total_mentions):
    """Specificity in (0, 1]: a gene mentioned with everything is down-weighted."""
    if total_mentions is None:
        return 0.5
    return 1.0 / math.log10(total_mentions + 10.0)


def path_chain_support(cache, drug, cancer, path):
    """Mechanism-chain literature support for one path. None if network-unavailable."""
    genes = [g for g in path.get("genes", []) if g]
    if not genes:
        return None
    tg = genes[0]            # drug's direct target
    dg = genes[-1]           # disease-associated gene (== tg for direct_target)
    h_dt, y_dt = count_recency(cache, drug, tg)                    # drug<->target
    h_gc, y_gc = count_recency(cache, dg, cancer, phrase_b=False)  # gene<->cancer
    h_tg, _ = count_recency(cache, tg)                            # target promiscuity
    h_dg, _ = count_recency(cache, dg)                            # bridge promiscuity
    if None in (h_dt, h_gc, h_tg, h_dg):
        return None
    co = math.log1p(h_dt) + math.log1p(h_gc)
    recency = 0.5 * (recency_weight(y_dt) + recency_weight(y_gc))
    spec = min(_spec_gene(h_tg), _spec_gene(h_dg))
    support = co * recency * spec
    return {
        "target_gene": tg, "disease_gene": dg,
        "hits_drug_target": h_dt, "hits_gene_cancer": h_gc,
        "gene_total_mentions": max(h_tg, h_dg),
        "recency": round(recency, 3), "specificity": round(spec, 3),
        "co_mention": round(co, 3), "chain_support": round(support, 4),
    }


def score_pair(cache, data, idx, dr, ds, drug_names, dis_names, want_contra=True):
    """Return structure-only + evidence-weighted mechanism scores and path detail."""
    drug = drug_names[dr]
    cancer = query_token(dis_names[ds])
    paths = mechanism_paths(data, idx, dr, ds, max_paths=8)
    struct = mechanism_score(paths)
    rec = {"drug": drug, "cancer": cancer, "struct_score": float(struct),
           "ew_chain": 0.0, "ew_full": 0.0, "n_paths": len(paths),
           "network_unavailable": False, "contra": 0,
           "hits_drug_cancer": None, "indication_factor": None, "paths": []}
    if not paths:
        return rec

    # Indication evidence (pair-level, used by the ew_full extension): is THIS drug
    # studied in THIS cancer at all? Loose cancer match (verbose labels).
    h_dc, _ = count_recency(cache, drug, cancer, phrase_b=False)
    if h_dc is None:
        rec["network_unavailable"] = True
        indication_factor = 1.0  # neutral on a network gap; do not zero it out
    else:
        rec["hits_drug_cancer"] = h_dc
        indication_factor = math.log1p(h_dc)
    rec["indication_factor"] = round(indication_factor, 4)

    best_ew_path = 0.0
    any_support = False
    for p in paths[:TOP_K_PATHS]:
        sup = path_chain_support(cache, drug, cancer, p)
        if sup is None:
            rec["network_unavailable"] = True
            continue
        any_support = True
        ew_path = float(p["score"]) * sup["chain_support"]
        best_ew_path = max(best_ew_path, ew_path)
        rec["paths"].append({
            "text": p["text"], "type": p["type"],
            "struct_score": round(float(p["score"]), 4),
            "ew_path": round(ew_path, 4), **sup,
        })

    contra = 0
    if want_contra and any_support:
        scan = contradiction_scan(cache, drug, cancer, per_query=8)
        contra = scan["contradicting"]
    penalty = 1.0 / (1.0 + CONTRA_ALPHA * contra)
    rec["contra"] = contra
    if any_support:
        # ew_chain: spec-faithful (drug-gene & gene-cancer co-mention, recency,
        # specificity, contradiction penalty). ew_full: + indication evidence
        # (drug-cancer co-mention), the term that separates shared-target negatives.
        rec["ew_chain"] = best_ew_path * penalty
        rec["ew_full"] = best_ew_path * indication_factor * penalty
    return rec


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    y = np.r_[np.ones_like(pos), np.zeros_like(neg)]
    s = np.r_[pos, neg]
    if len(np.unique(s)) == 1:
        return 0.5
    return float(roc_auc_score(y, s))


def find_hub_demotions(true_recs, max_examples=6):
    """Paths that are the STRUCTURE-top path for their pair but get demoted by evidence.

    These are coincidental hub paths: high structural score, but the bridge gene is
    promiscuous and/or the drug is never co-mentioned with the target, so the
    evidence weight pushes a different (or no) path to the top.
    """
    examples = []
    for r in true_recs:
        if len(r["paths"]) < 2:
            continue
        by_struct = sorted(r["paths"], key=lambda p: -p["struct_score"])
        by_ew = sorted(r["paths"], key=lambda p: -p["ew_path"])
        top_struct = by_struct[0]
        top_ew = by_ew[0]
        # demoted: the structure-top path is NOT the evidence-top path, and the
        # structure-top path looks hub-like (huge gene mention total or no drug
        # co-mention with its target).
        if top_struct["text"] == top_ew["text"]:
            continue
        hublike = (top_struct["gene_total_mentions"] >= 5000
                   or top_struct["hits_drug_target"] == 0)
        if not hublike:
            continue
        examples.append({
            "drug": r["drug"], "cancer": r["cancer"],
            "demoted_path": top_struct["text"],
            "demoted_struct_score": top_struct["struct_score"],
            "demoted_ew_path": top_struct["ew_path"],
            "demoted_gene_total_mentions": top_struct["gene_total_mentions"],
            "demoted_hits_drug_target": top_struct["hits_drug_target"],
            "promoted_path": top_ew["text"],
            "promoted_ew_path": top_ew["ew_path"],
            "promoted_hits_drug_target": top_ew["hits_drug_target"],
        })
    examples.sort(key=lambda e: -e["demoted_gene_total_mentions"])
    return examples[:max_examples]


def find_coincidental_negative_demotions(st_recs, true_median_ew, max_examples=6):
    """Shared-target hard negatives whose high-structure path is killed by evidence.

    A shared-target negative gets a strong structural direct-target path (it shares
    the target gene with the true drug), but the DRUG is barely studied in that
    cancer, so the indication factor + chain support collapse its evidence-weighted
    score far below the true-pair median. This is the coincidental-path demotion the
    retrieval layer is meant to deliver.
    """
    out = []
    for r in st_recs:
        if r["struct_score"] < 3.0 or not r["paths"]:
            continue  # need a strong (direct-target) structural path to be a coincidence
        if r["ew_full"] >= max(true_median_ew, 1e-9):
            continue  # not demoted relative to a typical true pair
        top = max(r["paths"], key=lambda p: p["struct_score"])
        out.append({
            "drug": r["drug"], "cancer": r["cancer"],
            "path": top["text"], "struct_score": top["struct_score"],
            "ew_full": round(r["ew_full"], 4),
            "hits_drug_cancer": r.get("hits_drug_cancer"),
            "hits_drug_target": top.get("hits_drug_target"),
        })
    out.sort(key=lambda e: (e["ew_full"], (e["hits_drug_cancer"] or 0)))
    return out[:max_examples]


def score_group(cache, data, idx, pairs, drug_names, dis_names, label, want_contra=True):
    recs = []
    for i, (dr, ds) in enumerate(pairs):
        recs.append(score_pair(cache, data, idx, dr, ds, drug_names, dis_names, want_contra))
        if (i + 1) % 10 == 0:
            cache.save()
            print(f"  [{label}] {i+1}/{len(pairs)} "
                  f"(cache {cache.stats['hits']}, live {cache.stats['live']}, "
                  f"err {cache.stats['errors']})")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="pairs per group")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--no-contra", action="store_true",
                    help="skip the contradiction penalty (ablation)")
    args = ap.parse_args()

    n = args.n or (10 if args.smoke else 50)
    suffix = "_smoke" if args.smoke else ""

    data, _ = load_primekg(with_features=False)
    idx = build_mech_index(data)
    drug_names = list(data[DRUG_TYPE].node_names)
    dis_names = list(data[DISEASE_TYPE].node_names)

    true_pairs, rand_pairs, st_pairs = sample_pairs(data, idx, n, n)
    print(f"true: {len(true_pairs)} | random: {len(rand_pairs)} | "
          f"shared_target: {len(st_pairs)}")

    cache = EPMCCache(CACHE_PATH, sleep=args.sleep)
    want_contra = not args.no_contra

    print("\nScoring TRUE pairs...")
    true_recs = score_group(cache, data, idx, true_pairs, drug_names, dis_names, "true", want_contra)
    print("Scoring RANDOM negatives (mostly no-path, cheap)...")
    rand_recs = score_group(cache, data, idx, rand_pairs, drug_names, dis_names, "random", want_contra)
    print("Scoring SHARED-TARGET hard negatives...")
    st_recs = score_group(cache, data, idx, st_pairs, drug_names, dis_names, "shared_target", want_contra)
    cache.save()

    def arr(recs, key):
        return np.array([r[key] for r in recs], float)

    s_true, ec_true, ef_true = arr(true_recs, "struct_score"), arr(true_recs, "ew_chain"), arr(true_recs, "ew_full")
    s_rand, ec_rand, ef_rand = arr(rand_recs, "struct_score"), arr(rand_recs, "ew_chain"), arr(rand_recs, "ew_full")
    s_st, ec_st, ef_st = arr(st_recs, "struct_score"), arr(st_recs, "ew_chain"), arr(st_recs, "ew_full")

    def comp(pos_s, neg_s, pos_ec, neg_ec, pos_ef, neg_ef):
        st_a = auroc(pos_s, neg_s)
        ec_a = auroc(pos_ec, neg_ec)
        ef_a = auroc(pos_ef, neg_ef)
        return {
            "structure_only": st_a,
            "evidence_chain": ec_a,
            "evidence_full": ef_a,
            "delta_chain": round(ec_a - st_a, 4),
            "delta_full": round(ef_a - st_a, 4),
        }

    results_auroc = {
        "true_vs_random": comp(s_true, s_rand, ec_true, ec_rand, ef_true, ef_rand),
        "true_vs_shared_target": comp(s_true, s_st, ec_true, ec_st, ef_true, ef_st),
    }

    hub_demotions = find_hub_demotions(true_recs)
    true_median_ew = float(np.median(ef_true)) if len(ef_true) else 0.0
    coincidental_demotions = find_coincidental_negative_demotions(st_recs, true_median_ew)

    network_blocked = (cache.stats["live"] == 0 and cache.stats["hits"] == 0
                       and cache.stats["errors"] > 0)
    any_unavailable = sum(r["network_unavailable"] for r in true_recs + st_recs)

    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "config": {"n_per_group": n, "top_k_paths": TOP_K_PATHS,
                   "contra_alpha": CONTRA_ALPHA, "contradiction_penalty": want_contra,
                   "sleep": args.sleep},
        "epmc_stats": cache.stats,
        "network_blocked": network_blocked,
        "pairs_with_network_gap": int(any_unavailable),
        "auroc": results_auroc,
        "group_means": {
            "struct": {"true": float(np.nanmean(s_true)), "random": float(np.nanmean(s_rand)),
                       "shared_target": float(np.nanmean(s_st))},
            "evidence_chain": {"true": float(np.nanmean(ec_true)),
                               "random": float(np.nanmean(ec_rand)),
                               "shared_target": float(np.nanmean(ec_st))},
            "evidence_full": {"true": float(np.nanmean(ef_true)),
                              "random": float(np.nanmean(ef_rand)),
                              "shared_target": float(np.nanmean(ef_st))},
        },
        "true_median_ew_score": round(true_median_ew, 4),
        "hub_demotions": hub_demotions,
        "coincidental_negative_demotions": coincidental_demotions,
        "true_pairs": true_recs,
        "random_pairs": rand_recs,
        "shared_target_pairs": st_recs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"evidence_weighted_graph{suffix}.json"
    out_md = RESULTS_DIR / f"evidence_weighted_graph{suffix}.md"
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(out_md, result)

    # ---- headline numbers ------------------------------------------------ #
    print("\n=== Evidence-weighted mechanism scoring ===")
    if network_blocked:
        print("NETWORK BLOCKED: no live calls and no cache; evidence weights are "
              "unavailable. Re-run with connectivity.")
    tvr, tvs = results_auroc["true_vs_random"], results_auroc["true_vs_shared_target"]
    print(f"true vs random        : structure {tvr['structure_only']:.3f} | "
          f"ev-chain {tvr['evidence_chain']:.3f} ({tvr['delta_chain']:+.3f}) | "
          f"ev-full {tvr['evidence_full']:.3f} ({tvr['delta_full']:+.3f})")
    print(f"true vs shared_target : structure {tvs['structure_only']:.3f} | "
          f"ev-chain {tvs['evidence_chain']:.3f} ({tvs['delta_chain']:+.3f}) | "
          f"ev-full {tvs['evidence_full']:.3f} ({tvs['delta_full']:+.3f})")
    print(f"\nWithin-pair hub paths demoted by evidence weighting: {len(hub_demotions)} examples")
    for e in hub_demotions[:3]:
        print(f"  DEMOTED  {e['demoted_path']}")
        print(f"           struct={e['demoted_struct_score']:.2f} ew_path={e['demoted_ew_path']:.4f} "
              f"gene_total_mentions={e['demoted_gene_total_mentions']} "
              f"drug-target hits={e['demoted_hits_drug_target']}")
        print(f"     -> promoted: {e['promoted_path']} (ew_path={e['promoted_ew_path']:.4f}, "
              f"drug-target hits={e['promoted_hits_drug_target']})")
    print(f"\nCoincidental shared-target paths demoted below the true-pair median "
          f"(ew {true_median_ew:.3f}): {len(coincidental_demotions)}")
    for e in coincidental_demotions[:3]:
        print(f"  COINCIDENCE  {e['path']}")
        print(f"               struct={e['struct_score']:.2f} -> ew_full={e['ew_full']:.4f} "
              f"| drug-cancer hits={e['hits_drug_cancer']} drug-target hits={e['hits_drug_target']}")
    print(f"\nsaved -> {out_json}\nsaved -> {out_md}")


def write_markdown(path, r):
    a = r["auroc"]
    tvr, tvs = a["true_vs_random"], a["true_vs_shared_target"]
    gm = r["group_means"]
    lines = [
        f"# Evidence-weighted mechanism scoring ({r['mode']})",
        "",
        f"_{r['timestamp']}_",
        "",
        "## What this shows",
        "",
        "The graph's structure-only mechanism score treats a coincidental "
        "drug->hub-gene->cancer chain the same as a real, literature-attested "
        "mechanism. Here we re-weight each mechanism path by Europe PMC evidence "
        "(co-mention strength of both links + recency + bridge-gene specificity, minus "
        "a contradiction penalty) and test whether that **load-bearing retrieval layer** "
        "sharpens separation -- especially against shared-target hard negatives, where "
        "the coincidental direct-target path is exactly the failure mode.",
        "",
    ]
    if r.get("network_blocked"):
        lines += ["> **Network was blocked and no cache was available** -- evidence "
                  "weights could not be computed. Re-run with connectivity.", ""]
    if r.get("pairs_with_network_gap"):
        lines += [f"> Note: {r['pairs_with_network_gap']} scored pairs had a partial "
                  "network gap on at least one path (those paths were skipped).", ""]
    lines += [
        "## Separation AUROC: structure-only vs evidence-weighted",
        "",
        "Two evidence scores: **ev-chain** is spec-faithful (drug-gene & gene-cancer "
        "co-mention x recency x bridge-gene specificity, minus a contradiction penalty). "
        "**ev-full** additionally multiplies by indication evidence -- whether the drug "
        "is co-mentioned with the cancer at all -- which is the term that can separate "
        "shared-target hard negatives (they share the mechanism chain by construction).",
        "",
        "| comparison | structure-only | ev-chain (delta) | ev-full (delta) |",
        "|---|---|---|---|",
        f"| true vs random | {tvr['structure_only']:.3f} | {tvr['evidence_chain']:.3f} "
        f"({tvr['delta_chain']:+.3f}) | {tvr['evidence_full']:.3f} ({tvr['delta_full']:+.3f}) |",
        f"| true vs shared-target (hard) | {tvs['structure_only']:.3f} | "
        f"{tvs['evidence_chain']:.3f} ({tvs['delta_chain']:+.3f}) | "
        f"{tvs['evidence_full']:.3f} ({tvs['delta_full']:+.3f}) |",
        "",
        f"Mean scores -- structure: true {gm['struct']['true']:.3f}, random "
        f"{gm['struct']['random']:.3f}, shared-target {gm['struct']['shared_target']:.3f}. "
        f"ev-full: true {gm['evidence_full']['true']:.3f}, random "
        f"{gm['evidence_full']['random']:.3f}, shared-target "
        f"{gm['evidence_full']['shared_target']:.3f}.",
        "",
        "## Coincidental hub paths demoted by evidence weighting",
        "",
        "Each row was the **top path by graph structure** for its (drug, cancer) pair "
        "but is NOT the top path after evidence weighting, because its bridge gene is a "
        "literature hub (mentioned with everything) and/or the drug is never co-mentioned "
        "with the target. These are the coincidences the retrieval layer is meant to catch.",
        "",
    ]
    if r["hub_demotions"]:
        lines += ["| pair | demoted path (structure-top) | gene total mentions | drug-target hits | "
                  "evidence-top path |",
                  "|---|---|---|---|---|"]
        for e in r["hub_demotions"]:
            lines.append(
                f"| {e['drug']} / {e['cancer']} | `{e['demoted_path']}` "
                f"(struct {e['demoted_struct_score']:.2f}, ew {e['demoted_ew_path']:.4f}) | "
                f"{e['demoted_gene_total_mentions']} | {e['demoted_hits_drug_target']} | "
                f"`{e['promoted_path']}` (ew {e['promoted_ew_path']:.4f}, "
                f"drug-target hits {e['promoted_hits_drug_target']}) |")
    else:
        lines.append("_No clear hub demotions in this sample._")
    lines += [
        "",
        "## Coincidental shared-target paths demoted below the true-pair bar",
        "",
        "These are shared-target HARD NEGATIVES: a drug that shares a target gene with "
        "the true drug for this cancer (so it inherits a strong structural direct-target "
        "path) but is **not** an indication. Evidence weighting collapses their score "
        f"below the true-pair median evidence-weighted score "
        f"({r.get('true_median_ew_score', 0)}), because the drug is barely co-mentioned "
        "with the cancer in the literature.",
        "",
    ]
    if r.get("coincidental_negative_demotions"):
        lines += ["| drug | cancer | coincidental structural path | struct | ew | "
                  "drug-cancer hits |",
                  "|---|---|---|---|---|---|"]
        for e in r["coincidental_negative_demotions"]:
            lines.append(
                f"| {e['drug']} | {e['cancer']} | `{e['path']}` | "
                f"{e['struct_score']:.2f} | {e['ew_full']:.4f} | {e['hits_drug_cancer']} |")
    else:
        lines.append("_No shared-target coincidences demoted in this sample._")
    lines += [
        "",
        "## Honest reading & caveats",
        "",
        f"- Sample is {r['config']['n_per_group']} pairs/group; AUROCs are indicative, "
        "not the full 400-pair headline. Random negatives mostly have no path, so the "
        "true-vs-random comparison is already near-saturated and has limited headroom; "
        "the **shared-target** comparison is where evidence weighting is supposed to help.",
        "- Co-mention counts are raw Europe PMC `hitCount`s: an exact-phrase pairing can "
        "still co-occur for non-mechanistic reasons, so the weight is a soft prior, not "
        "proof of mechanism.",
        "- Gene-symbol ambiguity (short symbols, gene/alias collisions) inflates some "
        "counts; the specificity term partly compensates but is imperfect.",
        "- The contradiction penalty reuses the lexical scan (see "
        "`contradiction_detector.py`) and inherits its noise.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
