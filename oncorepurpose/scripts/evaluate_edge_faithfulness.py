#!/usr/bin/env python
"""Counterfactual edge-faithfulness evaluation for the GNN's MECHANISM head.

Question: when the joint GNN names the curated DrugMechDB bridge gene for a held-out
(drug, cancer) pair, is that prediction FAITHFUL to the curated mechanism-of-action
(MOA) edges -- i.e. does it actually rely on the drug->target `drug_protein` edge(s)
of the bridge gene, rather than on spurious graph structure?

We answer it with a counterfactual edge-removal test on a trained model (NO
retraining; we just recompute embeddings with `model.encode(modified_base)`):

  FULL          intact graph: score/rank of the true bridge gene g for (d, c).
  REMOVE-MOA    delete ONLY this pair's drug->target MOA edge(s) (d, g) for the
                curated bridge gene(s) (both directions). A faithful model's score
                for g should DROP a lot (comprehensiveness).
  REMOVE-RANDOM control: delete a matched number of OTHER `drug_protein` edges of
                the SAME drug d (degree-matched random top-up if too few). A
                faithful model should drop LESS than under REMOVE-MOA.
  SUFFICIENCY   (optional) keep ONLY the MOA edges among d's targets; a faithful
                model should retain most of g's score.

Headline = faithfulness CONTRAST: (score/rank drop under MOA removal) vs (drop
under random removal), with a paired Wilcoxon test, the fraction of instances where
MOA-removal hurts more than random removal, and a separation AUROC. We report it
whichever way it falls -- if random removal hurts as much as MOA removal, the
mechanism reasoning is NOT demonstrably faithful and we say so.

Run:
  PYTHONPATH=. python scripts/evaluate_edge_faithfulness.py --smoke
  PYTHONPATH=. python scripts/evaluate_edge_faithfulness.py
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import wilcoxon

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.evaluation.mechanism_supervision import (
    DegreeMatchedDecoys, build_drugmechdb_drug_symbols, build_mech_examples,
    symbol_to_gene_index,
)
from oncorepurpose.evaluation.splits import make_split
from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn_joint
from oncorepurpose.interpret.mechanism_paths import build_mech_index
from oncorepurpose.models import HeteroGNN

# Reuse the building blocks from the existing mechanism-recovery experiment WITHOUT
# editing or importing it as a package module (it defines main() under __main__).
_EMR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_mechanism_recovery.py")
_spec = importlib.util.spec_from_file_location("_emr_helpers", _EMR_PATH)
emr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emr)
blind_base = emr.blind_base            # remove drug_protein (d,g) edges, both dirs
drug2prot_from = emr.drug2prot_from    # drug_idx -> set(gene_idx) targets
gnn_scores = emr.gnn_scores            # all-gene mechanism scores for (d, c)
positives = emr.positives

GENE = "gene_protein"
TARGET = (DRUG_TYPE, "indication", DISEASE_TYPE)
DP = (DRUG_TYPE, "drug_protein", GENE)


def rank_of(scores: torch.Tensor, g: int) -> int:
    """0-indexed rank of gene g (number of genes scoring strictly higher)."""
    return int((scores > scores[g]).sum().item())


def all_dp_edges(base):
    ei = base[DP].edge_index
    return list(zip(ei[0].tolist(), ei[1].tolist()))


def sample_random_removal(di, non_moa, n_rm, dp_pool, moa_set, rng):
    """Pick n_rm drug_protein edges to delete as a matched control.

    Prefer OTHER targets of the SAME drug d; if too few, top up with global random
    drug_protein edges (degree-matched in expectation). Returns (remove_set, n_fallback).
    """
    rm = set()
    pool = list(non_moa)
    if len(pool) >= n_rm:
        chosen = rng.choice(len(pool), size=n_rm, replace=False)
        return {(di, pool[int(i)]) for i in chosen}, 0
    rm = {(di, g) for g in pool}
    need = n_rm - len(pool)
    fb = need
    tries = 0
    while need > 0 and tries < need * 200 + 50:
        e = tuple(dp_pool[int(rng.integers(0, len(dp_pool)))])
        if e not in rm and not (e[0] == di and e[1] in moa_set):
            rm.add(e)
            need -= 1
        tries += 1
    return rm, fb


def run_seed(data, idx, dmdb, sym2gidx, seed, device, epochs, lam, n_random, max_pairs):
    set_all_seeds(seed)
    split = make_split(data, TARGET, "inductive_cold_dst", seed=seed, restrict_oncology=True)
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}
    num_genes = int(data[GENE].num_nodes)

    tr_pos = positives(split.train_label_index, split.train_label)
    te_pos = positives(split.test_label_index, split.test_label)
    mech_tr = build_mech_examples(data, tr_pos, dmdb, sym2gidx)
    mech_te = build_mech_examples(data, te_pos, dmdb, sym2gidx)
    if not mech_te.pairs:
        return None

    decoys = DegreeMatchedDecoys(idx["prot_drug_deg"], num_genes, seed=seed)
    joint = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims).to(device)
    joint = train_gnn_joint(joint, split, device, mech_tr, decoys, lam=lam, epochs=epochs)
    joint.eval()

    base = split.base
    d2p = drug2prot_from(base)
    dp_pool = all_dp_edges(base)
    rng = np.random.default_rng(1000 + seed)

    pairs = list(mech_te.pairs)
    n_all_pairs = len(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]

    with torch.no_grad():
        z_full = joint.encode(base)

    instances = []
    n_skipped = 0
    for (di, ci, genes) in pairs:
        targets = d2p.get(di, set())
        moa = sorted(set(genes) & targets)              # bridge genes that ARE edges
        if not moa:
            n_skipped += 1                              # no MOA edge to ablate
            continue
        non_moa = sorted(targets - set(moa))
        n_rm = len(moa)

        s_full = gnn_scores(joint, z_full, di, ci, num_genes, device)

        # --- REMOVE-MOA (comprehensiveness) ---
        with torch.no_grad():
            z_moa = joint.encode(blind_base(base, {(di, g) for g in moa}))
        s_moa = gnn_scores(joint, z_moa, di, ci, num_genes, device)

        # --- REMOVE-RANDOM (control), averaged over draws ---
        rand_drops = defaultdict(list)
        rand_rank = defaultdict(list)
        fb_total = 0
        for _ in range(n_random):
            rm, fb = sample_random_removal(di, non_moa, n_rm, dp_pool, set(moa), rng)
            fb_total += fb
            with torch.no_grad():
                z_r = joint.encode(blind_base(base, rm))
            s_r = gnn_scores(joint, z_r, di, ci, num_genes, device)
            for g in moa:
                rand_drops[g].append(float(s_full[g] - s_r[g]))
                rand_rank[g].append(rank_of(s_r, g) - rank_of(s_full, g))

        # --- SUFFICIENCY (optional): keep ONLY MOA edges among d's targets ---
        s_suff = None
        if non_moa:
            with torch.no_grad():
                z_s = joint.encode(blind_base(base, {(di, g) for g in non_moa}))
            s_suff = gnn_scores(joint, z_s, di, ci, num_genes, device)

        for g in moa:
            rk_full = rank_of(s_full, g)
            instances.append({
                "drug": int(di), "dis": int(ci), "gene": int(g),
                "n_drug_targets": len(targets), "n_moa_removed": n_rm,
                "n_random_fallback": int(fb_total),
                "score_full": float(s_full[g]), "rank_full": rk_full,
                "score_moa": float(s_moa[g]), "rank_moa": rank_of(s_moa, g),
                "moa_score_drop": float(s_full[g] - s_moa[g]),
                "moa_rank_degrade": rank_of(s_moa, g) - rk_full,
                "rand_score_drop": float(np.mean(rand_drops[g])),
                "rand_rank_degrade": float(np.mean(rand_rank[g])),
                "suff_score": (float(s_suff[g]) if s_suff is not None else float(s_full[g])),
                "suff_score_drop": (float(s_full[g] - s_suff[g]) if s_suff is not None else 0.0),
            })
    return {"n_test_pairs": n_all_pairs, "n_pairs_evaluated": len(pairs),
            "n_pairs_no_moa_edge": n_skipped, "n_instances": len(instances),
            "instances": instances}


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def seed_summary(seed_out):
    """Per-seed aggregate metrics over its instances."""
    inst = seed_out["instances"]
    moa_s = [r["moa_score_drop"] for r in inst]
    rand_s = [r["rand_score_drop"] for r in inst]
    moa_r = [r["moa_rank_degrade"] for r in inst]
    rand_r = [r["rand_rank_degrade"] for r in inst]
    faithful = [1.0 if r["moa_score_drop"] > r["rand_score_drop"] else 0.0 for r in inst]
    return {
        "n_instances": len(inst),
        "mean_moa_score_drop": _mean(moa_s),
        "mean_rand_score_drop": _mean(rand_s),
        "mean_score_contrast": _mean([a - b for a, b in zip(moa_s, rand_s)]),
        "mean_moa_rank_degrade": _mean(moa_r),
        "mean_rand_rank_degrade": _mean(rand_r),
        "mean_rank_contrast": _mean([a - b for a, b in zip(moa_r, rand_r)]),
        "fraction_faithful": _mean(faithful),
        "mean_suff_score_drop": _mean([r["suff_score_drop"] for r in inst]),
    }


def mean_std(vals):
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return {"mean": None, "std": None, "values": []}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "values": [round(float(v), 4) for v in vals]}


def pooled_stats(all_inst):
    moa_s = np.array([r["moa_score_drop"] for r in all_inst], dtype=float)
    rand_s = np.array([r["rand_score_drop"] for r in all_inst], dtype=float)
    n = len(all_inst)
    frac_faithful = float(np.mean(moa_s > rand_s)) if n else float("nan")
    # Paired Wilcoxon: is MOA-removal drop > random-removal drop, per instance?
    p_val, stat = None, None
    diff = moa_s - rand_s
    if n >= 1 and np.any(diff != 0):
        try:
            res = wilcoxon(moa_s, rand_s, alternative="greater", zero_method="wilcox")
            stat, p_val = float(res.statistic), float(res.pvalue)
        except Exception as e:  # pragma: no cover
            p_val = f"wilcoxon_failed: {e}"
    # Separation AUROC: can the drop magnitude tell MOA-removal from random-removal?
    auroc = None
    if n:
        try:
            from sklearn.metrics import roc_auc_score
            y = np.concatenate([np.ones(n), np.zeros(n)])
            s = np.concatenate([moa_s, rand_s])
            auroc = float(roc_auc_score(y, s))
        except Exception as e:  # pragma: no cover
            auroc = f"auroc_failed: {e}"
    return {
        "n_instances": n,
        "mean_moa_score_drop": _mean(moa_s.tolist()),
        "mean_rand_score_drop": _mean(rand_s.tolist()),
        "mean_score_contrast": _mean(diff.tolist()),
        "median_moa_score_drop": float(np.median(moa_s)) if n else None,
        "median_rand_score_drop": float(np.median(rand_s)) if n else None,
        "fraction_faithful": frac_faithful,
        "wilcoxon_stat": stat, "wilcoxon_p_greater": p_val,
        "separation_auroc": auroc,
        "mean_moa_rank_degrade": _mean([r["moa_rank_degrade"] for r in all_inst]),
        "mean_rand_rank_degrade": _mean([r["rand_rank_degrade"] for r in all_inst]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-random", type=int, default=3, help="random-removal draws per pair")
    ap.add_argument("--max-pairs", type=int, default=0, help="cap pairs per seed (0=all)")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.n_random, args.max_pairs = [0], 12, 1, 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_primekg(with_features=True)
    idx = build_mech_index(data)
    dmdb = build_drugmechdb_drug_symbols()
    sym2gidx = symbol_to_gene_index(data)
    print(f"DrugMechDB drugs mapped: {len(dmdb)} | device {device}")

    max_pairs = args.max_pairs if args.max_pairs > 0 else None
    seeds_out, per_seed, all_inst = [], [], []
    for s in args.seeds:
        r = run_seed(data, idx, dmdb, sym2gidx, s, device, args.epochs, args.lam,
                     args.n_random, max_pairs)
        if r is None:
            print(f"seed {s}: no covered held-out pairs"); continue
        ss = seed_summary(r)
        print(f"\nseed {s}: test_pairs={r['n_test_pairs']} evaluated={r['n_pairs_evaluated']} "
              f"no_moa_edge={r['n_pairs_no_moa_edge']} instances={r['n_instances']}")
        print(f"  MOA score drop = {ss['mean_moa_score_drop']:.3f} | "
              f"random score drop = {ss['mean_rand_score_drop']:.3f} | "
              f"contrast = {ss['mean_score_contrast']:.3f}")
        print(f"  MOA rank degrade = {ss['mean_moa_rank_degrade']:.1f} | "
              f"random rank degrade = {ss['mean_rand_rank_degrade']:.1f}")
        print(f"  fraction faithful = {ss['fraction_faithful']:.3f} | "
              f"sufficiency score drop = {ss['mean_suff_score_drop']:.3f}")
        seeds_out.append({k: v for k, v in r.items() if k != "instances"} | {"seed": s})
        per_seed.append({"seed": s, **ss})
        all_inst.extend(r["instances"])

    if not all_inst:
        print("No faithfulness instances produced (no covered held-out pairs with MOA edges).")
        return

    pooled = pooled_stats(all_inst)
    headline = {
        "mean_moa_score_drop": mean_std([p["mean_moa_score_drop"] for p in per_seed]),
        "mean_rand_score_drop": mean_std([p["mean_rand_score_drop"] for p in per_seed]),
        "mean_score_contrast": mean_std([p["mean_score_contrast"] for p in per_seed]),
        "mean_moa_rank_degrade": mean_std([p["mean_moa_rank_degrade"] for p in per_seed]),
        "mean_rand_rank_degrade": mean_std([p["mean_rand_rank_degrade"] for p in per_seed]),
        "fraction_faithful": mean_std([p["fraction_faithful"] for p in per_seed]),
        "mean_suff_score_drop": mean_std([p["mean_suff_score_drop"] for p in per_seed]),
    }
    contrast = pooled["mean_score_contrast"]
    # A ratio is only meaningful when the random-removal drop is clearly positive;
    # near-zero/negative denominators make it explode/flip sign, so report None then.
    den = pooled["mean_rand_score_drop"]
    ratio = (pooled["mean_moa_score_drop"] / den
             if isinstance(den, float) and den > 1e-3 else None)
    faithful_verdict = (
        pooled["fraction_faithful"] is not None and pooled["fraction_faithful"] > 0.5
        and isinstance(pooled["wilcoxon_p_greater"], float) and pooled["wilcoxon_p_greater"] < 0.05
        and contrast > 0
    )

    summary = {
        "config": {"seeds": args.seeds, "epochs": args.epochs, "lam": args.lam,
                   "n_random_draws": args.n_random, "max_pairs": args.max_pairs,
                   "device": str(device), "smoke": args.smoke},
        "method": ("Counterfactual edge removal on a trained joint GNN (no retraining). "
                   "Per held-out (drug, cancer, bridge-gene) instance: compare the bridge "
                   "gene's mechanism-score/rank drop when its curated MOA drug_protein "
                   "edge(s) are deleted (REMOVE-MOA) vs when a matched number of the same "
                   "drug's OTHER target edges are deleted (REMOVE-RANDOM)."),
        "n_instances": len(all_inst),
        "per_seed": per_seed,
        "seeds_meta": seeds_out,
        "headline_seed_mean_std": headline,
        "pooled": pooled,
        "score_drop_ratio_moa_over_random": ratio,
        "faithful_verdict": bool(faithful_verdict),
    }
    summary["interpretation"] = build_interpretation(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "edge_faithfulness_eval.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    out_md = os.path.join(RESULTS_DIR, "edge_faithfulness_eval.md")
    with open(out_md, "w") as f:
        f.write(build_markdown(summary))

    print("\n" + summary["interpretation"])
    print(f"Saved -> {out_json}")
    print(f"Saved -> {out_md}")


def build_interpretation(summary):
    p = summary["pooled"]
    pv = p["wilcoxon_p_greater"]
    pv_s = f"{pv:.2e}" if isinstance(pv, float) else str(pv)
    verdict = ("FAITHFUL: MOA-edge removal degrades the bridge gene's score "
               "significantly more than removing other target edges of the same drug."
               if summary["faithful_verdict"] else
               "NOT clearly faithful: MOA-edge removal does not hurt significantly more "
               "than the random-edge control.")
    return (
        f"{verdict} Pooled over {p['n_instances']} held-out (drug,cancer,bridge-gene) "
        f"instances: mean score drop under MOA removal = {p['mean_moa_score_drop']:.3f} "
        f"vs random removal = {p['mean_rand_score_drop']:.3f} (contrast = "
        f"{p['mean_score_contrast']:.3f}); fraction of instances where MOA removal hurts "
        f"more = {p['fraction_faithful']:.3f}; paired Wilcoxon p (MOA>random) = {pv_s}; "
        f"separation AUROC = {p['separation_auroc'] if not isinstance(p['separation_auroc'], float) else round(p['separation_auroc'], 3)}. "
        f"Mean rank degradation: MOA = {p['mean_moa_rank_degrade']:.1f} vs random = "
        f"{p['mean_rand_rank_degrade']:.1f} positions.")


def build_markdown(summary):
    h = summary["headline_seed_mean_std"]
    p = summary["pooled"]
    c = summary["config"]

    def ms(d):
        if d["mean"] is None:
            return "n/a"
        return f"{d['mean']:.3f} ± {d['std']:.3f}"

    pv = p["wilcoxon_p_greater"]
    pv_s = f"{pv:.2e}" if isinstance(pv, float) else str(pv)
    auroc = p["separation_auroc"]
    auroc_s = f"{auroc:.3f}" if isinstance(auroc, float) else str(auroc)
    ratio = summary["score_drop_ratio_moa_over_random"]
    ratio_s = (f"{ratio:.1f}x" if isinstance(ratio, float)
               else "n/a (random drop ≈ 0)")
    verdict = "FAITHFUL" if summary["faithful_verdict"] else "NOT clearly faithful"

    return f"""# Counterfactual Edge-Faithfulness of GNN Mechanism Predictions

**Verdict: {verdict}**

Does the joint GNN, when it names the curated DrugMechDB bridge gene for a held-out
(drug, cancer) pair, actually rely on that gene's curated mechanism-of-action (MOA)
`drug_protein` edge -- or on spurious structure? We test this with a counterfactual
edge-removal experiment on the trained model (no retraining): recompute embeddings
after deleting edges and measure the bridge gene's mechanism score/rank drop.

- Setup: inductive cold-disease (oncology) split; joint GNN (link BCE + InfoNCE
  mechanism aux), seeds {c['seeds']}, {c['epochs']} epochs, {c['n_random_draws']}
  random-removal draws per pair.
- Sample: **{summary['n_instances']}** held-out (drug, cancer, bridge-gene) instances
  (covered pairs whose curated bridge gene is an actual `drug_protein` edge).

## Headline faithfulness contrast (mechanism score, logits)

| Condition | Mean score drop (seed mean±std) |
|---|---|
| REMOVE-MOA (this pair's MOA edge) | {ms(h['mean_moa_score_drop'])} |
| REMOVE-RANDOM (matched other targets of same drug) | {ms(h['mean_rand_score_drop'])} |
| **Contrast (MOA − random)** | **{ms(h['mean_score_contrast'])}** |

- Ratio of mean drops (MOA / random): **{ratio_s}**
- Fraction of instances where MOA removal hurts more than random: **{ms(h['fraction_faithful'])}**
- Paired Wilcoxon (MOA drop > random drop), pooled: **p = {pv_s}**
- Separation AUROC (MOA vs random drops): **{auroc_s}**
- Mean rank degradation (positions): MOA = {ms(h['mean_moa_rank_degrade'])} vs random = {ms(h['mean_rand_rank_degrade'])}
- Sufficiency (keep only MOA edges) score drop: {ms(h['mean_suff_score_drop'])} (smaller = more sufficient)

## Interpretation

{summary['interpretation']}

### Caveats
- "Score" is the mechanism-head logit; drops are in logit space (monotonic, not
  probability). Ranks are over all `gene_protein` nodes.
- Faithfulness is measured only on covered pairs whose bridge gene is a real
  `drug_protein` edge (others have no MOA edge to ablate); this is the population
  where the question is well-posed, not all predictions.
- Removing an edge changes both endpoints' embeddings; the random control isolates
  "is it THIS edge" by deleting the same drug's other target edges (with degree-
  matched global top-up when a drug has too few other targets).
- Small held-out covered set; treat as a faithfulness probe, not a population estimate.
"""


if __name__ == "__main__":
    main()
