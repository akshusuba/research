#!/usr/bin/env python
"""Hard-negative stress test for the OncoEvidence mechanism-path signal.

The headline result (`scripts/evaluate_mechanism.py`) is that the graph
mechanism score separates true oncology indications from RANDOM (drug, cancer)
pairs at AUROC ~0.879. Reviewers object that random negatives are too easy: a
high-degree-hub drug paired with a high-degree-hub cancer will trivially have
*some* path, so the classifier might be learning "popularity", not "mechanism".

This script (read-only w.r.t. the existing code) recomputes the separation
AUROC against progressively *harder, fairer* negatives, all restricted to
oncology diseases and all excluding known therapeutic pairs (indication /
contraindication / off-label, both directions):

  1. random          -- random (drug, oncology disease).            [baseline]
  2. degree_matched  -- negative whose drug- and disease-degree bins match the
                        true pair's (decile-binned over ALL incident edges),
                        so degree/hubness is held fixed.
  3. oncology_drug   -- (drug indicated for SOME cancer, a different cancer):
                        real oncology drugs vs cancers they do NOT treat.
  4. shared_target   -- (drug, cancer) where the drug shares >=1 target protein
                        with the true drug for that cancer but lacks the
                        indication. Skipped gracefully if infeasible.

It also runs three mechanism-score ABLATIONS on the random setup:
  - full              -- the normal `mechanism_score` (max path score).
  - template_only     -- best path TYPE bonus only (direct=3/ppi=2/pathway=1).
  - no_direct_target  -- `mechanism_score` over paths with type != direct_target.

Run:
    PYTHONPATH=. python scripts/evaluate_hard_negatives.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from oncorepurpose.config import (
    DISEASE_TYPE, DRUG_TYPE, INDICATION_REL, RESULTS_DIR,
)
from oncorepurpose.datasets import load_primekg
from oncorepurpose.interpret.mechanism_paths import (
    build_mech_index, mechanism_paths, mechanism_score,
)
from oncorepurpose.interpret.paths import _known_pairs

N_TRUE = 400
N_NEG = 400
SEED = 0
MAX_PATHS = 25  # generous so template_only / no_direct_target see every path type

# Path-type bonus used by the template_only ablation.
TYPE_BONUS = {"direct_target": 3.0, "ppi": 2.0, "shared_pathway": 1.0}


# --------------------------------------------------------------------------- #
# Graph helpers
# --------------------------------------------------------------------------- #
def oncology_disease_indices(data):
    store = data[DISEASE_TYPE]
    if "is_oncology" in store:
        return set(torch.nonzero(store.is_oncology, as_tuple=False).flatten().tolist())
    return set(range(int(store.num_nodes)))


def node_degrees(data, node_type, num_nodes):
    """Total incident-edge count per node across ALL edge types (both directions)."""
    deg = torch.zeros(num_nodes, dtype=torch.long)
    for et in data.edge_types:
        s, _, d = et
        ei = data[et].edge_index
        if s == node_type:
            deg += torch.bincount(ei[0], minlength=num_nodes)
        if d == node_type:
            deg += torch.bincount(ei[1], minlength=num_nodes)
    return deg.numpy()


def decile_bins(degrees, members):
    """Assign each member node to a degree decile (0..9); return {bin: [nodes]}.

    Quantile edges are computed over the degrees of `members` so the bins are
    balanced within the relevant population.
    """
    members = list(members)
    vals = degrees[members]
    edges = np.quantile(vals, np.linspace(0, 1, 11))
    # interior edges only; np.digitize with right=False puts max into bin 10 -> clamp.
    interior = edges[1:-1]
    bin_of = {}
    bucket = {b: [] for b in range(10)}
    for n in members:
        b = int(np.digitize(degrees[n], interior, right=False))
        b = min(b, 9)
        bin_of[n] = b
        bucket[b].append(n)
    return bin_of, bucket


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def pair_scores(data, idx, pairs):
    """Return per-pair (full, template_only, no_direct_target, any_path) arrays.

    All four are derived from a single `mechanism_paths` call per pair.
    """
    full, templ, no_dt, any_path = [], [], [], []
    for dr, ds in pairs:
        paths = mechanism_paths(data, idx, dr, ds, max_paths=MAX_PATHS)
        full.append(mechanism_score(paths))
        templ.append(max((TYPE_BONUS.get(p["type"], 0.0) for p in paths), default=0.0))
        filt = [p for p in paths if p["type"] != "direct_target"]
        no_dt.append(mechanism_score(filt))
        any_path.append(1.0 if paths else 0.0)
    return (np.array(full, float), np.array(templ, float),
            np.array(no_dt, float), np.array(any_path, float))


def auroc(pos, neg):
    if len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones_like(pos), np.zeros_like(neg)]
    s = np.r_[pos, neg]
    return float(roc_auc_score(y, s))


# --------------------------------------------------------------------------- #
# Negative samplers (all exclude `known`, dedup, restrict diseases to oncology)
# --------------------------------------------------------------------------- #
def sample_random(rng, true_pairs, known, num_drugs, onco_list):
    out, seen = [], set()
    tries = 0
    while len(out) < N_NEG and tries < N_NEG * 50:
        tries += 1
        dr = rng.randrange(num_drugs)
        ds = rng.choice(onco_list)
        if (dr, ds) in known or (dr, ds) in seen:
            continue
        seen.add((dr, ds))
        out.append((dr, ds))
    return out


def sample_degree_matched(rng, true_pairs, known, drug_bin, drug_bucket,
                          dis_bin, dis_bucket):
    """One negative per true pair, matching the true drug- and disease-degree bin."""
    out, seen = [], set()
    for dr_t, ds_t in true_pairs:
        db = drug_bin.get(dr_t)
        sb = dis_bin.get(ds_t)
        drug_pool = drug_bucket.get(db, [])
        dis_pool = dis_bucket.get(sb, [])
        if not drug_pool or not dis_pool:
            continue
        for _ in range(200):
            dr = rng.choice(drug_pool)
            ds = rng.choice(dis_pool)
            if (dr, ds) in known or (dr, ds) in seen:
                continue
            seen.add((dr, ds))
            out.append((dr, ds))
            break
    return out


def sample_oncology_drug(rng, true_pairs, known, onco_drugs, onco_list):
    """(drug indicated for SOME cancer, a cancer it does not treat)."""
    onco_drugs = list(onco_drugs)
    out, seen = [], set()
    tries = 0
    while len(out) < N_NEG and tries < N_NEG * 50:
        tries += 1
        dr = rng.choice(onco_drugs)
        ds = rng.choice(onco_list)
        if (dr, ds) in known or (dr, ds) in seen:
            continue
        seen.add((dr, ds))
        out.append((dr, ds))
    return out


def sample_shared_target(rng, true_pairs, known, idx, onco_set):
    """For each true (drug, cancer), a different drug sharing >=1 target, same cancer."""
    drug2prot = idx["drug2prot"]
    prot2drug = idx["prot2drug"]
    out, seen = [], set()
    for dr_t, ds in true_pairs:
        if ds not in onco_set:
            continue
        targets = drug2prot.get(dr_t, set())
        if not targets:
            continue
        cands = set()
        for p in targets:
            cands |= prot2drug.get(p, set())
        cands.discard(dr_t)
        cands = [c for c in cands
                 if (c, ds) not in known and (c, ds) not in seen]
        if not cands:
            continue
        dr = rng.choice(cands)
        seen.add((dr, ds))
        out.append((dr, ds))
    return out


# --------------------------------------------------------------------------- #
def main():
    data, targets = load_primekg(with_features=False)
    idx = build_mech_index(data)

    onco_set = oncology_disease_indices(data)
    onco_list = sorted(onco_set)
    known = _known_pairs(data)
    num_drugs = int(data[DRUG_TYPE].num_nodes)
    num_dis = int(data[DISEASE_TYPE].num_nodes)

    ind_et = targets["indication"]
    ei = data[ind_et].edge_index
    # Orient so [0]=drug, [1]=disease regardless of stored direction.
    if ind_et[0] == DRUG_TYPE:
        ind_drug, ind_dis = ei[0].tolist(), ei[1].tolist()
    else:
        ind_drug, ind_dis = ei[1].tolist(), ei[0].tolist()

    # The SAME 400 true oncology indication pairs as the baseline (seed 0).
    rng = random.Random(SEED)
    true_pairs = [(dr, ds) for dr, ds in zip(ind_drug, ind_dis) if ds in onco_set]
    rng.shuffle(true_pairs)
    true_pairs = true_pairs[:N_TRUE]

    onco_drugs = {dr for dr, ds in zip(ind_drug, ind_dis) if ds in onco_set}

    print(f"true oncology indication pairs: {len(true_pairs)}")
    print(f"oncology diseases: {len(onco_list)} | "
          f"oncology-indicated drugs: {len(onco_drugs)} | "
          f"known therapeutic pairs excluded: {len(known)}")

    # Degree deciles (over ALL incident edges).
    drug_deg = node_degrees(data, DRUG_TYPE, num_drugs)
    dis_deg = node_degrees(data, DISEASE_TYPE, num_dis)
    drug_bin, drug_bucket = decile_bins(drug_deg, range(num_drugs))
    dis_bin, dis_bucket = decile_bins(dis_deg, onco_set)

    # True-pair scores (shared positive set for every comparison + ablations).
    s_true, t_true, ndt_true, ap_true = pair_scores(data, idx, true_pairs)

    # --- Build negatives -------------------------------------------------- #
    negsets = {
        "random": sample_random(
            random.Random(SEED + 1), true_pairs, known, num_drugs, onco_list),
        "degree_matched": sample_degree_matched(
            random.Random(SEED + 2), true_pairs, known,
            drug_bin, drug_bucket, dis_bin, dis_bucket),
        "oncology_drug": sample_oncology_drug(
            random.Random(SEED + 3), true_pairs, known, onco_drugs, onco_list),
    }
    try:
        st = sample_shared_target(
            random.Random(SEED + 4), true_pairs, known, idx, onco_set)
        if st:
            negsets["shared_target"] = st
        else:
            print("shared_target: no candidates found -> skipped")
    except Exception as exc:  # graceful skip
        print(f"shared_target: skipped ({exc})")

    # --- Negative-type table ---------------------------------------------- #
    neg_results = {}
    rows = []
    for name, negs in negsets.items():
        s_neg, _, _, ap_neg = pair_scores(data, idx, negs)
        au = auroc(s_true, s_neg)
        neg_mean = float(s_neg.mean()) if len(s_neg) else float("nan")
        neg_ap = float(ap_neg.mean()) if len(ap_neg) else float("nan")
        neg_results[name] = {
            "n_neg": len(negs), "auroc": au,
            "neg_mean_score": neg_mean, "neg_any_path_rate": neg_ap,
        }
        rows.append((name, len(negs), au, neg_mean, neg_ap))

    print("\n=== Mechanism-score separation vs harder negatives "
          "(positives: 400 true oncology indications) ===")
    print(f"{'negative_type':<16} {'n_neg':>6} {'AUROC':>8} "
          f"{'neg_mean_score':>15} {'neg_any_path_rate':>18}")
    for name, n, au, ms, ap in rows:
        print(f"{name:<16} {n:>6} {au:>8.3f} {ms:>15.3f} {ap:>18.1%}")
    print(f"(reference) true: mean_score={s_true.mean():.3f}  "
          f"any_path_rate={ap_true.mean():.1%}")

    # --- Ablations on the random setup ------------------------------------ #
    s_rand, t_rand, ndt_rand, _ = pair_scores(data, idx, negsets["random"])
    ablations = {
        "full": auroc(s_true, s_rand),
        "template_only": auroc(t_true, t_rand),
        "no_direct_target": auroc(ndt_true, ndt_rand),
    }
    print("\n=== Mechanism-score ablations (true vs random) ===")
    print(f"{'ablation':<18} {'AUROC':>8}")
    for name, au in ablations.items():
        print(f"{name:<18} {au:>8.3f}")

    # --- Interpretation --------------------------------------------------- #
    finite = {k: v["auroc"] for k, v in neg_results.items()
              if v["auroc"] == v["auroc"]}  # drop NaN
    hardest = min(finite, key=finite.get) if finite else None
    interp = (
        f"Baseline random AUROC={neg_results.get('random', {}).get('auroc', float('nan')):.3f}. "
        f"Hardest negative = {hardest} (AUROC={finite.get(hardest, float('nan')):.3f}). "
        f"Indirect-only mechanism (no_direct_target) AUROC="
        f"{ablations['no_direct_target']:.3f}; template-only AUROC="
        f"{ablations['template_only']:.3f}. "
        "The 0.879 separation holds up if harder negatives stay well above 0.5 and "
        "the no_direct_target ablation remains clearly above chance; it is degree-/"
        "direct-target-driven to the extent those values collapse toward 0.5."
    )

    out = {
        "n_true": len(true_pairs),
        "seed": SEED,
        "max_paths": MAX_PATHS,
        "negative_types": neg_results,
        "ablations": {k: {"auroc": v} for k, v in ablations.items()},
        "interpretation": interp,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "hard_negatives_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print(f"\nInterpretation: {interp}")


if __name__ == "__main__":
    main()
