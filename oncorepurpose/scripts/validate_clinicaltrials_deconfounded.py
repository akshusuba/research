#!/usr/bin/env python
"""Popularity-DECONFOUNDED ClinicalTrials.gov validation for OncoEvidence.

The first orthogonal check (``validate_clinicaltrials.py``) found that the raw
model score tracks trial existence (AUROC ~0.68) but is heavily confounded by drug
popularity: a few broadly-indicated drugs (e.g. folic acid) score high for EVERY
cancer and are trialed for EVERY cancer, so the signal is "popular drug" rather
than "right drug for this cancer". This script removes that confound by
CONDITIONING ON THE DRUG.

Within-drug stratified test
---------------------------
For each focus drug we ask: among the cancers where this drug is a NOVEL (not
already-known) candidate, does the model score the cancers that actually have a
real interventional trial ABOVE the cancers that do not? Because every comparison
is between two cancers for the SAME drug, the drug's overall popularity cancels
out -- a drug that is trialed everywhere contributes both hits and non-hits and
cannot inflate the score.

  - Stratified AUROC = (sum over drugs of concordant cancer-pairs) /
                       (sum over drugs of comparable cancer-pairs),
    where a comparable pair is (trial-cancer, non-trial-cancer) for one drug and
    it is concordant if score(trial-cancer) > score(non-trial-cancer).
  - Significance via a within-stratum label permutation test (shuffle each drug's
    trial labels across its cancers; AUROC = 0.5 under the null).
  - A complementary sign test on each drug's "is the top-scored cancer a trial?"

This reuses the cached model scores (data/ctval_model_scores.json) produced by
validate_clinicaltrials.py and the cached ClinicalTrials.gov client, so re-runs
are cheap.

Run:
    PYTHONPATH=. python scripts/validate_clinicaltrials_deconfounded.py --smoke
    PYTHONPATH=. python scripts/validate_clinicaltrials_deconfounded.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.validate_clinicaltrials import CTClient, clean_disease  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCORES_CACHE = REPO / "data" / "ctval_model_scores.json"
CACHE_PATH = REPO / "data" / "clinicaltrials_cache.json"


def load_scores():
    if not SCORES_CACHE.exists():
        raise SystemExit(
            "data/ctval_model_scores.json not found. Run "
            "`PYTHONPATH=. python scripts/validate_clinicaltrials.py` first to train "
            "the deployment GNN and cache per-(drug, cancer) scores.")
    d = json.loads(SCORES_CACHE.read_text())
    drug_names = d["drug_names"]
    diseases = d["diseases"]  # [{idx, name, clean}]
    ranked = {int(k): v for k, v in d["ranked"].items()}
    return drug_names, diseases, ranked


def stratified_auroc(strata):
    """strata: list of (scores[np], labels[np 0/1]). Returns pooled within-stratum AUROC."""
    conc = comp = 0.0
    for scores, labels in strata:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for sp in pos:
            comp += len(neg)
            conc += np.sum(sp > neg) + 0.5 * np.sum(sp == neg)
    return (conc / comp) if comp else float("nan"), int(comp)


def permutation_p_strata(strata, observed, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    usable = [(s, l) for (s, l) in strata if l.sum() > 0 and l.sum() < len(l)]
    if not usable or not np.isfinite(observed):
        return None
    ge = 0
    for _ in range(n_perm):
        perm = [(s, rng.permutation(l)) for (s, l) in usable]
        a, _ = stratified_auroc(perm)
        if np.isfinite(a) and a >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def two_way_residual(S, mask, iters=200):
    """Additive two-way (drug + cancer) fixed-effect fit on observed entries; return
    the residual R = S - mu - drug_effect - cancer_effect. Handles missing cells.

    Removing both an additive drug effect (drug popularity) AND an additive cancer
    effect (cancer popularity) leaves only the drug x cancer INTERACTION: how much
    the model elevates THIS pair beyond what either marginal would predict."""
    mu = float(np.nanmean(S))
    a = np.zeros(S.shape[0])
    b = np.zeros(S.shape[1])
    Sc = np.where(mask, S, np.nan)
    for _ in range(iters):
        a = np.nan_to_num(np.nanmean(Sc - mu - b[None, :], axis=1))
        b = np.nan_to_num(np.nanmean(Sc - mu - a[:, None], axis=0))
    R = Sc - mu - a[:, None] - b[None, :]
    return R, mu, a, b


def interaction_auroc(R, H, mask):
    """AUROC of trial-hit vs the doubly-centered residual score over observed cells."""
    from sklearn.metrics import roc_auc_score
    r = R[mask]
    h = H[mask].astype(int)
    if 0 < h.sum() < len(h):
        return float(roc_auc_score(h, r))
    return float("nan")


def interaction_perm_p(R, H, mask, observed, n_perm=2000, seed=0):
    """Permutation null that respects structure: shuffle each drug-row's residuals
    across that row's observed cancers, breaking any true drug x cancer alignment
    while preserving every row's residual multiset and the full hit matrix."""
    from sklearn.metrics import roc_auc_score
    if not np.isfinite(observed):
        return None
    rng = np.random.default_rng(seed)
    h = H[mask].astype(int)
    ge = 0
    for _ in range(n_perm):
        Rp = R.copy()
        for i in range(R.shape[0]):
            idx = np.where(mask[i])[0]
            if len(idx) > 1:
                Rp[i, idx] = R[i, rng.permutation(idx)]
        a = roc_auc_score(h, Rp[mask])
        if a >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-disease-top", type=int, default=8,
                    help="take this many top-by-score novel drugs per cancer to form the focus-drug pool")
    ap.add_argument("--max-drugs", type=int, default=70, help="cap distinct focus drugs")
    ap.add_argument("--sleep", type=float, default=0.34)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        args.per_disease_top = 3
        args.max_drugs = 12
        args.n_perm = 500
        out_json = REPO / "results" / "clinicaltrials_deconfounded_smoke.json"
    else:
        out_json = REPO / "results" / "clinicaltrials_deconfounded.json"
    out_md = out_json.with_suffix(".md")

    drug_names, diseases, ranked = load_scores()
    dz_clean = {d["idx"]: d["clean"] for d in diseases}
    dz_list = [d["idx"] for d in diseases if d["idx"] in ranked]

    # score(drug, disease) for every NOVEL pair (known indications are absent from `ranked`)
    score_of = {}
    for dz in dz_list:
        for di, sc, _lift in ranked[dz]:
            score_of[(int(di), dz)] = float(sc)

    # Focus drugs = union of top-by-score novel drugs per cancer (the ones we'd surface).
    focus = []
    seen = set()
    by_score = {dz: sorted(ranked[dz], key=lambda r: r[1], reverse=True) for dz in dz_list}
    # round-robin so the pool is not dominated by one cancer
    for rank in range(args.per_disease_top):
        for dz in dz_list:
            if rank < len(by_score[dz]):
                di = int(by_score[dz][rank][0])
                if di not in seen:
                    seen.add(di); focus.append(di)
    focus = focus[: args.max_drugs]
    print(f"focus drugs: {len(focus)} | cancers: {len(dz_list)} | "
          f"max queries: {len(focus) * len(dz_list)}")

    # Query ClinicalTrials.gov for every (focus drug, cancer) novel pair.
    client = CTClient(CACHE_PATH, sleep=args.sleep)
    pairs = []
    for di in focus:
        for dz in dz_list:
            if (di, dz) not in score_of:   # known indication -> not a novel prediction
                continue
            res = client.query(drug_names[di], dz_clean[dz])
            pairs.append({"drug_idx": di, "disease_idx": dz,
                          "drug": str(drug_names[di]), "cancer": dz_clean[dz],
                          "score": score_of[(di, dz)],
                          "trial_hit": int(res["hit"]), "trial_total": int(res["total"]),
                          "example_nct": (res.get("examples") or [{}])[0].get("nct")})
    client.save()
    print(f"queried pairs: {len(pairs)} | network calls: {client.n_network_calls}")

    # Build per-drug strata (for the within-drug AUROC, controls drug popularity).
    by_drug = defaultdict(list)
    for p in pairs:
        by_drug[p["drug_idx"]].append(p)
    strata = []
    usable_drugs = 0
    for di, items in by_drug.items():
        scores = np.array([it["score"] for it in items], float)
        labels = np.array([it["trial_hit"] for it in items], int)
        strata.append((scores, labels))
        if 0 < labels.sum() < len(labels):
            usable_drugs += 1
    auroc, n_comparable = stratified_auroc(strata)
    p_perm = permutation_p_strata(strata, auroc, n_perm=args.n_perm, seed=args.seed)

    # ---- PRIMARY: doubly-centered drug x cancer INTERACTION test --------------
    # Removes BOTH drug popularity (row effect) AND cancer popularity (column
    # effect); only the interaction survives. This answers the sharp question:
    # does the model elevate the SPECIFIC (drug, cancer) pairs that have trials,
    # beyond what the drug's and the cancer's overall popularity already explain?
    drug_ids = sorted({p["drug_idx"] for p in pairs})
    canc_ids = sorted({p["disease_idx"] for p in pairs})
    di_pos = {d: i for i, d in enumerate(drug_ids)}
    ci_pos = {c: i for i, c in enumerate(canc_ids)}
    S = np.full((len(drug_ids), len(canc_ids)), np.nan)
    H = np.full_like(S, np.nan)
    NCT = {}
    for p in pairs:
        i, j = di_pos[p["drug_idx"]], ci_pos[p["disease_idx"]]
        S[i, j] = p["score"]; H[i, j] = p["trial_hit"]
        NCT[(i, j)] = p.get("example_nct")
    mask = ~np.isnan(S)
    R, mu, row_eff, col_eff = two_way_residual(S, mask)
    inter_auroc = interaction_auroc(R, H, mask)
    inter_p = interaction_perm_p(R, H, mask, inter_auroc, n_perm=args.n_perm, seed=args.seed)

    # Interaction-based clean examples: trial pairs the model elevated MOST above
    # both baselines (high positive residual). These cannot be popularity artifacts.
    inter_examples = []
    name_by_drug = {di_pos[d]: drug_names[d] for d in drug_ids}
    name_by_canc = {ci_pos[c]: dz_clean[c] for c in canc_ids}
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if mask[i, j] and H[i, j] == 1 and np.isfinite(R[i, j]):
                inter_examples.append({
                    "drug": name_by_drug[i], "cancer": name_by_canc[j],
                    "score": round(float(S[i, j]), 4),
                    "interaction_residual": round(float(R[i, j]), 4),
                    "example_nct": NCT.get((i, j)),
                })
    inter_examples.sort(key=lambda e: e["interaction_residual"], reverse=True)

    # Sign test: per usable drug, is the TOP-scored cancer a trial-hit more often
    # than the drug's own base rate? (Within-drug, popularity-free.)
    top_hit = base_rate = n_drugs_for_sign = 0
    per_drug_top = []
    for di, items in by_drug.items():
        labels = np.array([it["trial_hit"] for it in items], int)
        if labels.sum() == 0 or len(items) < 3:
            continue
        n_drugs_for_sign += 1
        top = max(items, key=lambda it: it["score"])
        top_hit += int(top["trial_hit"])
        base_rate += labels.mean()
        per_drug_top.append({
            "drug": top["drug"], "top_cancer": top["cancer"],
            "top_score": round(top["score"], 4), "top_is_trial": int(top["trial_hit"]),
            "drug_base_rate": round(float(labels.mean()), 3),
            "example_nct": top["example_nct"],
        })
    top_hit_rate = top_hit / n_drugs_for_sign if n_drugs_for_sign else float("nan")
    mean_base_rate = base_rate / n_drugs_for_sign if n_drugs_for_sign else float("nan")

    # Binomial tail: P(>= top_hit successes | base = mean_base_rate) for the sign-style test.
    sign_p = None
    if n_drugs_for_sign and np.isfinite(mean_base_rate) and 0 < mean_base_rate < 1:
        from scipy.stats import binomtest
        sign_p = float(binomtest(top_hit, n_drugs_for_sign, mean_base_rate,
                                 alternative="greater").pvalue)

    overall_hit_frac = float(np.mean([p["trial_hit"] for p in pairs])) if pairs else float("nan")
    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "design": "two-way fixed-effect drug x cancer interaction (drug + cancer popularity removed)",
        "config": {"per_disease_top": args.per_disease_top, "max_drugs": args.max_drugs,
                   "n_perm": args.n_perm, "seed": args.seed},
        "counts": {"focus_drugs": len(focus), "cancers": len(dz_list),
                   "pairs_evaluated": len(pairs),
                   "usable_drugs_for_within_drug": usable_drugs,
                   "comparable_within_drug_pairs": n_comparable,
                   "overall_pair_hit_fraction": overall_hit_frac},
        "interaction_auroc": inter_auroc,
        "interaction_auroc_perm_p": inter_p,
        "within_drug_auroc": auroc,
        "within_drug_auroc_perm_p": p_perm,
        "top_cancer_sign_test": {
            "n_drugs": n_drugs_for_sign, "top_is_trial_count": top_hit,
            "top_hit_rate": top_hit_rate, "mean_drug_base_rate": mean_base_rate,
            "binomial_p_greater": sign_p,
        },
        "interaction_examples": inter_examples[:15],
        "per_drug_top": per_drug_top,
    }
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(out_md, result)
    print(f"\nInteraction (drug+cancer de-confounded) AUROC = {inter_auroc:.3f} "
          f"(perm p = {inter_p if inter_p is None else f'{inter_p:.3g}'})")
    print(f"Within-drug stratified AUROC = {auroc:.3f} "
          f"(perm p = {p_perm if p_perm is None else f'{p_perm:.3g}'}; "
          f"{usable_drugs} drugs, {n_comparable} comparable cancer-pairs)")
    print(f"saved -> {out_json}\nsaved -> {out_md}")


def write_markdown(path, r):
    c = r["counts"]
    ia = r["interaction_auroc"]
    ip = r["interaction_auroc_perm_p"]
    wa = r["within_drug_auroc"]
    wp = r["within_drug_auroc_perm_p"]
    st = r["top_cancer_sign_test"]
    ip_s = "n/a" if ip is None else f"{ip:.3g}"
    wp_s = "n/a" if wp is None else f"{wp:.3g}"
    is_pos = np.isfinite(ia) and ia > 0.5 and ip is not None and ip < 0.05
    is_null = np.isfinite(ia) and ia <= 0.55
    verdict = ("a genuine drug x cancer signal that neither drug nor cancer popularity can explain"
               if is_pos
               else "no clean interaction signal once both popularity effects are removed"
               if is_null else "a weak/indicative interaction signal")
    tail = ("After subtracting both popularity baselines, the model still scores the specific "
            "drug-cancer combinations that have real trials above the combinations that do not."
            if is_pos else
            "Once both popularity baselines are subtracted, the model's per-pair scores no longer "
            "track which specific drug-cancer combinations have trials. In other words, the raw "
            "deployment scores carry drug- and cancer-level popularity, but not pairwise "
            "repurposing specificity that this external registry can corroborate. We report this "
            "as a characterized limitation rather than a win, consistent with the project's "
            "honest-evaluation stance; the time-split prospective analysis (Finding 5) is where "
            "specific predictive signal does show up.")
    lines = [
        f"# ClinicalTrials.gov validation, popularity-deconfounded ({r['mode']})",
        "",
        f"_{r['timestamp']}_",
        "",
        "## Why this test",
        "",
        "The raw-score orthogonal check (see `clinicaltrials_validation.md`) is confounded by "
        "popularity on two sides: some drugs are trialed for every cancer, and some cancers are "
        "trialed with every drug. A positive raw AUROC can therefore reflect 'popular drug' or "
        "'popular cancer' rather than 'right drug for this cancer'. We remove **both** effects and "
        "test only what is left: the drug x cancer **interaction**.",
        "",
        "## Headline -- interaction test (primary)",
        "",
        "We fit an additive two-way model score(drug, cancer) = mu + drug-effect + cancer-effect "
        "on the evaluated pairs and take the residual. The residual is how much the model elevates "
        "a *specific* pair beyond what the drug's and the cancer's overall popularity predict. "
        "We then ask whether that residual predicts a real interventional trial.",
        "",
        f"- **Interaction AUROC = {ia:.3f}** (0.5 = no signal beyond popularity), structure-aware "
        f"permutation p = **{ip_s}**.",
        f"- Evaluated over **{c['pairs_evaluated']}** novel (drug, cancer) pairs spanning "
        f"**{c['focus_drugs']}** focus drugs x **{c['cancers']}** cancers (overall pair hit-fraction "
        f"{c['overall_pair_hit_fraction']:.3f}).",
        "",
        f"**Read:** {verdict}. {tail}",
        "",
        "## Supporting -- within-drug stratified AUROC (controls drug popularity only)",
        "",
        f"- **Within-drug AUROC = {wa:.3f}**, permutation p = **{wp_s}**, from "
        f"**{c['usable_drugs_for_within_drug']}** drugs over "
        f"**{c['comparable_within_drug_pairs']}** within-drug cancer pairs.",
        f"- Top-cancer sign test: for **{st['top_is_trial_count']}/{st['n_drugs']}** drugs the "
        f"single highest-scored novel cancer has a real trial, vs a per-drug base rate of "
        f"{st['mean_drug_base_rate']:.3f}"
        + (f" (binomial p = {st['binomial_p_greater']:.3g})." if st['binomial_p_greater'] is not None else "."),
        "(This one controls drug popularity but not cancer popularity, which is why the interaction "
        "test above is the cleaner number.)",
        "",
        "## Clean interaction wins (trial pairs the model elevated most above BOTH baselines)",
        "",
        "| drug | cancer | model score | interaction residual | example NCT |",
        "|---|---|---|---|---|",
    ]
    for e in r["interaction_examples"]:
        lines.append(
            f"| {e['drug']} | {e['cancer']} | {e['score']:.3f} | "
            f"{e['interaction_residual']:+.4f} | {e['example_nct'] or ''} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- Trial existence is a plausibility signal, not efficacy.",
        "- Name matching uses ClinicalTrials.gov free-text + synonym search (some misses / loose matches).",
        "- The focus-drug pool is the model's own top-scored novel candidates, so this tests "
        "whether, among drugs the model likes, it points at the right cancer; it does not "
        "re-test candidate selection itself.",
        "- The interaction residual is an additive de-confounder; strong multiplicative popularity "
        "effects could leave minor residual structure, so treat the magnitude as indicative.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
