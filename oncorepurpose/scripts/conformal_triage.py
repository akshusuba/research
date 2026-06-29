#!/usr/bin/env python
"""Split-conformal calibration for the OncoEvidence triage (deliverable 1).

The deployment ranker (a transductive HeteroGNN on PrimeKG indication edges)
emits a sigmoid link score per (drug, disease) pair. A raw score is not a
calibrated confidence and gives no principled way to decide *when to abstain*.
This script wraps the ranker in **split-conformal prediction** so that:

  * every candidate gets a calibrated confidence (a conformal p-value: how
    typical its score is among genuine indications), and
  * the triage ABSTAINS on any candidate whose confidence falls below the level
    needed to guarantee a target marginal coverage of true indications
    (default 90%).

Method (standard inductive / split conformal, calibrated on POSITIVES):
  1. Train the GNN on the training indication edges only (message passing +
     supervision); a small held-out fold is used for early stopping.
  2. A disjoint fold of known positives (never seen in training/early-stopping)
     is split into a CALIBRATION set and a TEST set; equal-sized sampled
     negatives accompany them.
  3. Nonconformity of a positive = 1 - score (a low-scoring true indication is
     "nonconforming"). For target coverage 1-alpha, the conformal threshold is
     the (1-alpha)(1+1/n) empirical quantile of the calibration nonconformities.
     Accept (do NOT abstain) iff nonconformity <= threshold, i.e. score >= s*.
  4. Empirical coverage is measured on the disjoint TEST positives; the
     abstention rate is measured on a realistic deployment pool (test positives
     + sampled negatives) and on the shortlist if present.

If ``results/repurposing_shortlist.json`` exists, every shortlisted candidate is
re-scored with THIS deployment model and emitted with its calibrated confidence
and accept/abstain decision to ``results/repurposing_shortlist_calibrated.json``.

Run:
    PYTHONPATH=. python scripts/conformal_triage.py            # full
    PYTHONPATH=. python scripts/conformal_triage.py --smoke     # fast sanity
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.evaluation.splits import transductive_split
from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn
from oncorepurpose.models import HeteroGNN

HIDDEN = 128
TARGET_COVERAGES = (0.80, 0.90, 0.95)
HEADLINE_COVERAGE = 0.90


# --------------------------------------------------------------------------- #
# Device / OOM handling
# --------------------------------------------------------------------------- #
def _is_oom(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def train_with_fallback(data, split, in_dims, epochs, patience, seed):
    """Train the GNN on CUDA if available, falling back to CPU on OOM."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for attempt_dev in ([dev, torch.device("cpu")] if dev.type == "cuda" else [dev]):
        try:
            set_all_seeds(seed)
            gnn = HeteroGNN(list(data.node_types), list(split.base.edge_types),
                            in_dims, hidden=HIDDEN, num_layers=2, dropout=0.3)
            gnn = train_gnn(gnn, split, attempt_dev, epochs=epochs, patience=patience)
            return gnn, attempt_dev
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc) and attempt_dev.type == "cuda":
                print(f"  [oom] CUDA OOM during training -> falling back to CPU")
                torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError("training failed on all devices")


@torch.no_grad()
def score_pairs(gnn, base, target, eli, device) -> np.ndarray:
    """Sigmoid link scores for an edge-label-index tensor [2, N]."""
    gnn.eval()
    z = gnn.encode(base)
    return torch.sigmoid(gnn.decode(z, target, eli)).cpu().numpy()


# --------------------------------------------------------------------------- #
# Conformal machinery (nonconformity = 1 - score, calibrated on positives)
# --------------------------------------------------------------------------- #
def conformal_threshold(calib_pos_scores: np.ndarray, target_coverage: float) -> float:
    """Score threshold s*: accept iff score >= s* to guarantee `target_coverage`.

    Uses the finite-sample-corrected (1-alpha)(1+1/n) quantile of the
    calibration nonconformities (1 - score).
    """
    alpha = 1.0 - target_coverage
    a = 1.0 - np.asarray(calib_pos_scores, dtype=float)
    n = a.size
    level = min(1.0, math.ceil((n + 1) * target_coverage) / n)
    q = float(np.quantile(a, level, method="higher"))
    return 1.0 - q  # back to score space


def conformal_pvalue(scores: np.ndarray, calib_pos_scores: np.ndarray) -> np.ndarray:
    """Conformal p-value per item: (1 + #calib positives no more conforming) / (n+1).

    With nonconformity A = 1 - score, A_calib >= A(x) <=> score_calib <= score(x).
    Higher p-value = the item's score is more typical of a genuine indication.
    """
    calib_sorted = np.sort(np.asarray(calib_pos_scores, dtype=float))
    n = calib_sorted.size
    # #calib positives with score <= score(x)
    le = np.searchsorted(calib_sorted, np.asarray(scores, dtype=float), side="right")
    return (1.0 + le) / (n + 1.0)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    epochs = args.epochs or (12 if args.smoke else 40)

    print(f"device(cuda available)={torch.cuda.is_available()} | epochs={epochs}")
    data, targets = load_primekg(with_features=True)
    target = targets["indication"]
    print(f"target edge type: {target}")
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}

    # Deployment split: train 60% / early-stop 10% / held-out 30%.
    # The 30% held-out fold (positives + sampled negatives) is disjoint from
    # training and feeds conformal calibration + coverage testing.
    set_all_seeds(args.seed)
    split = transductive_split(data, target, val_frac=0.1, test_frac=0.3,
                               neg_ratio=1.0, seed=args.seed)
    print(f"split positives: train={split.info['train_pos']} "
          f"val(early-stop)={split.info['val_pos']} held-out={split.info['test_pos']}")

    gnn, used_dev = train_with_fallback(data, split, in_dims, epochs, args.patience, args.seed)
    print(f"trained deployment GNN on {used_dev}")

    # Score the held-out fold; separate positives / negatives.
    held_scores = score_pairs(gnn, split.base, target, split.test_label_index, used_dev)
    held_labels = split.test_label.cpu().numpy().astype(int)
    pos_scores = held_scores[held_labels == 1]
    neg_scores = held_scores[held_labels == 0]
    held_auroc = float(roc_auc_score(held_labels, held_scores))
    print(f"held-out AUROC (score quality, context): {held_auroc:.3f} "
          f"| n_pos={pos_scores.size} n_neg={neg_scores.size}")

    # Split positives + negatives into disjoint calibration / test halves.
    rng = np.random.default_rng(args.seed)
    def halve(arr):
        idx = rng.permutation(arr.size)
        h = arr.size // 2
        return arr[idx[:h]], arr[idx[h:]]
    calib_pos, test_pos = halve(pos_scores)
    calib_neg, test_neg = halve(neg_scores)
    print(f"calibration positives={calib_pos.size} | test positives={test_pos.size}")

    # Per target coverage: threshold, empirical coverage on TEST positives,
    # abstention on a realistic pool (test positives + test negatives).
    eval_pool = np.concatenate([test_pos, test_neg])
    eval_pool_is_pos = np.concatenate([np.ones_like(test_pos), np.zeros_like(test_neg)]).astype(bool)

    coverage_table = []
    for cov in TARGET_COVERAGES:
        s_star = conformal_threshold(calib_pos, cov)
        emp_cov = float(np.mean(test_pos >= s_star))            # P(accept | true positive)
        accepted_pool = eval_pool >= s_star
        abst_rate = float(np.mean(~accepted_pool))               # fraction abstained in pool
        neg_abst = float(np.mean(test_neg < s_star)) if test_neg.size else float("nan")
        coverage_table.append({
            "target_coverage": cov,
            "score_threshold": s_star,
            "empirical_coverage_test_positives": emp_cov,
            "abstention_rate_pool": abst_rate,
            "negative_abstention_rate": neg_abst,
            "n_accepted_pool": int(accepted_pool.sum()),
            "n_pool": int(eval_pool.size),
        })
        print(f"  target {cov:.2f}: s*={s_star:.4f} "
              f"emp_coverage={emp_cov:.3f} abstain(pool)={abst_rate:.3f} "
              f"neg_abstain={neg_abst:.3f}")

    headline_row = next(r for r in coverage_table if r["target_coverage"] == HEADLINE_COVERAGE)

    # ----- Calibrate the shortlist (if it exists) ------------------------- #
    shortlist_path = RESULTS_DIR / "repurposing_shortlist.json"
    shortlist_out = None
    shortlist_summary = None
    if shortlist_path.exists():
        print(f"\nCalibrating shortlist: {shortlist_path}")
        sl = json.loads(shortlist_path.read_text())
        drug_names = list(data[DRUG_TYPE].node_names)
        dis_names = list(data[DISEASE_TYPE].node_names)
        drug2i = {n: i for i, n in enumerate(drug_names)}
        dis2i = {n: i for i, n in enumerate(dis_names)}

        alpha = 1.0 - HEADLINE_COVERAGE
        s_star = headline_row["score_threshold"]
        cand_drug_idx, cand_dis_idx, flat_refs = [], [], []
        for entry in sl.get("shortlist", []):
            dname = entry.get("disease")
            di = dis2i.get(dname)
            for cand in entry.get("candidates", []):
                ri = drug2i.get(cand.get("drug"))
                if ri is None or di is None:
                    flat_refs.append((cand, None))
                    continue
                flat_refs.append((cand, len(cand_drug_idx)))
                cand_drug_idx.append(ri)
                cand_dis_idx.append(di)

        if cand_drug_idx:
            eli = torch.tensor([cand_drug_idx, cand_dis_idx], dtype=torch.long)
            cand_scores = score_pairs(gnn, split.base, target, eli, used_dev)
            cand_pvals = conformal_pvalue(cand_scores, calib_pos)
        else:
            cand_scores = np.array([])
            cand_pvals = np.array([])

        n_accept = n_abstain = 0
        out_shortlist = []
        for entry in sl.get("shortlist", []):
            new_cands = []
            for cand in entry.get("candidates", []):
                ref = next((r for c, r in flat_refs if c is cand), None)
                cc = dict(cand)
                if ref is None:
                    cc["conformal"] = {"status": "unscored (node not found)"}
                else:
                    sc = float(cand_scores[ref])
                    pv = float(cand_pvals[ref])
                    decision = "accept" if pv >= alpha else "abstain"
                    cc["conformal"] = {
                        "deployment_score": sc,
                        "calibrated_confidence": pv,   # conformal p-value
                        "score_threshold": s_star,
                        "target_coverage": HEADLINE_COVERAGE,
                        "triage": decision,
                    }
                    n_accept += decision == "accept"
                    n_abstain += decision == "abstain"
                new_cands.append(cc)
            out_shortlist.append({"disease": entry.get("disease"), "candidates": new_cands})

        shortlist_out = {
            "target": sl.get("target"),
            "calibration": {
                "target_coverage": HEADLINE_COVERAGE,
                "score_threshold": s_star,
                "n_calibration_positives": int(calib_pos.size),
                "note": ("calibrated_confidence is the split-conformal p-value "
                         "(typicality of the score among true indications); "
                         "triage=abstain when confidence < alpha=1-coverage."),
            },
            "shortlist": out_shortlist,
        }
        total = n_accept + n_abstain
        shortlist_summary = {
            "n_candidates": total,
            "n_accept": n_accept,
            "n_abstain": n_abstain,
            "abstention_rate": (n_abstain / total) if total else None,
        }
        cal_path = RESULTS_DIR / "repurposing_shortlist_calibrated.json"
        cal_path.write_text(json.dumps(shortlist_out, indent=2))
        print(f"  shortlist candidates: {total} | accept={n_accept} abstain={n_abstain}")
        print(f"  saved -> {cal_path}")
    else:
        print("\nNo repurposing_shortlist.json found; skipping shortlist calibration.")

    # ----- Persist results ------------------------------------------------- #
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "dataset": "PrimeKG",
        "target_edge_type": list(target),
        "device": str(used_dev),
        "config": {"epochs": epochs, "patience": args.patience, "seed": args.seed,
                   "hidden": HIDDEN, "val_frac": 0.1, "test_frac": 0.3,
                   "target_coverages": list(TARGET_COVERAGES)},
        "split_positives": {"train": split.info["train_pos"],
                            "early_stop": split.info["val_pos"],
                            "held_out": split.info["test_pos"]},
        "held_out_score_auroc": held_auroc,
        "n_calibration_positives": int(calib_pos.size),
        "n_test_positives": int(test_pos.size),
        "n_test_negatives": int(test_neg.size),
        "coverage_table": coverage_table,
        "headline_coverage": HEADLINE_COVERAGE,
        "headline": (
            f"At target coverage {HEADLINE_COVERAGE:.0%}, split-conformal triage "
            f"achieves empirical coverage "
            f"{headline_row['empirical_coverage_test_positives']:.1%} on held-out "
            f"true indications (score threshold {headline_row['score_threshold']:.3f}), "
            f"abstaining on {headline_row['abstention_rate_pool']:.1%} of a "
            f"positives+negatives pool and {headline_row['negative_abstention_rate']:.1%} "
            f"of negatives."
        ),
        "shortlist_calibration": shortlist_summary,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "conformal_triage.json"
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(RESULTS_DIR / "conformal_triage.md", result)
    print(f"\n{result['headline']}")
    print(f"saved -> {out_json}")
    print(f"saved -> {RESULTS_DIR / 'conformal_triage.md'}")


def write_markdown(path, r):
    lines = [
        "# OncoEvidence: split-conformal triage calibration",
        "",
        f"_{r['timestamp_utc']}_  ·  mode: {r['mode']}  ·  device: {r['device']}",
        "",
        "Wraps the deployment ranker (transductive HeteroGNN on PrimeKG indication "
        "edges) in split-conformal prediction so each candidate gets a calibrated "
        "confidence and the triage can **abstain** below a coverage target.",
        "",
        "## Headline",
        "",
        r["headline"],
        "",
        "## Setup",
        "",
        f"- Positives split: train={r['split_positives']['train']}, "
        f"early-stop={r['split_positives']['early_stop']}, "
        f"held-out={r['split_positives']['held_out']} (held-out is disjoint from "
        "training/early-stopping).",
        f"- Calibration positives: {r['n_calibration_positives']}; "
        f"test positives: {r['n_test_positives']}; test negatives: {r['n_test_negatives']}.",
        f"- Held-out raw-score AUROC (context, not the conformal claim): "
        f"{r['held_out_score_auroc']:.3f}.",
        "- Nonconformity = 1 - sigmoid link score, calibrated on POSITIVES "
        "(coverage is a guarantee about true indications).",
        "",
        "## Coverage / abstention by target",
        "",
        "| Target coverage | Score threshold | Empirical coverage (test +) | "
        "Abstention (pool) | Negative abstention |",
        "|---|---|---|---|---|",
    ]
    for row in r["coverage_table"]:
        lines.append(
            f"| {row['target_coverage']:.0%} | {row['score_threshold']:.3f} | "
            f"{row['empirical_coverage_test_positives']:.1%} | "
            f"{row['abstention_rate_pool']:.1%} | {row['negative_abstention_rate']:.1%} |"
        )
    lines += ["", "## Shortlist calibration", ""]
    sc = r.get("shortlist_calibration")
    if sc:
        lines.append(
            f"Re-scored {sc['n_candidates']} shortlist candidates with the deployment "
            f"model: **{sc['n_accept']} accepted, {sc['n_abstain']} abstained** "
            f"(abstention rate {sc['abstention_rate']:.1%}) at "
            f"{r['headline_coverage']:.0%} target coverage. See "
            "`repurposing_shortlist_calibrated.json`."
        )
    else:
        lines.append("_No shortlist found; calibration skipped._")
    lines += [
        "",
        "## Honest read & caveats",
        "",
        "- Conformal coverage is a *marginal* finite-sample guarantee under "
        "exchangeability of the calibration and test positives; both are random "
        "holdouts of the same PrimeKG indication edges, so exchangeability is "
        "reasonable but the guarantee transfers to *novel* candidates only insofar "
        "as they are exchangeable with known indications (they may not be).",
        "- Negatives are sampled (assumed-negative) drug–disease pairs, so the "
        "negative-abstention rate is indicative, not a true specificity.",
        "- The calibration is only as good as the ranker; conformal makes the "
        "abstention decision *honest*, it does not improve ranking.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
