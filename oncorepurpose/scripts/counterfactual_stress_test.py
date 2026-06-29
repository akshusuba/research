#!/usr/bin/env python
"""Counterfactual Mechanism Stress Test (OncoEvidence HEADLINE evaluation).

OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided
Cancer Drug Repurposing. The novelty here is the EVALUATION, not a new ranker:
we ask whether the model ranks a drug for the RIGHT biological reason -- i.e. is
the proposed mechanism path causally important, biologically specific, and
literature-supported -- rather than merely "remove the real edge, score drops"
(the existing edge-faithfulness probe, Finding 6).

For a sample of known oncology (drug, cancer) pairs that have a curated
(DrugMechDB) or graph-extractable mechanism path
(drug -> target -> ... -> cancer-gene -> cancer), we run four counterfactual
tests and quantify each:

  (1) Target-edge ablation (causal importance). Remove the key drug->target MOA
      edge and measure the mechanism-score drop, vs removing a matched random
      edge of the SAME drug_protein type. Reuses the edge-faithfulness logic.
      Headline: score contrast + paired Wilcoxon + fraction faithful + AUROC.

  (2) Wrong-target substitution (specificity). Replace the true target with a
      plausible-but-wrong DECOY target (same node type, drug-degree-matched, NOT
      associated with the disease). Score the fake mechanism through the trained
      mechanism head. A faithful model must NOT prefer the decoy MOA.
      Headline: rejection rate (true scores above decoy) + true-vs-decoy AUROC.

  (3) Decoy-path swap + literature verifier (literature support). Keep the same
      drug and cancer but swap the real bridge gene for a decoy bridge gene; run
      the lexical MOA verifier on Europe PMC literature for the true vs decoy
      bridge and check it supports the true path and not the decoy.
      Headline: supported(/weak) rate, true vs decoy. (LLM step would strengthen
      this; no API key here, so the lexical/MOA-rubric path is used.)

  (4) True MOA vs PLAUSIBLE hard negatives (counterfactual specificity). Reuse
      the shared-target / oncology-drug / degree-matched hard negatives and frame
      the graph mechanism-score separation as a counterfactual specificity test.
      Headline: separation AUROC per negative class (reproduces ~0.887 random,
      ~0.609 shared-target from existing results).

Honest framing: this MOVES THE NEEDLE only if the model demonstrably (a) relies
on the true MOA edge (test 1), (b) rejects degree-matched fake targets (test 2),
(c) finds literature for the true bridge but not the decoy (test 3), and (d) the
mechanism signal survives plausible, non-random negatives (test 4). We report
each number whichever way it falls.

Run:
  PYTHONPATH=. python scripts/counterfactual_stress_test.py --smoke
  PYTHONPATH=. python scripts/counterfactual_stress_test.py
"""
import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.evaluation.mechanism_supervision import (
    DegreeMatchedDecoys, build_drugmechdb_drug_symbols, build_mech_examples,
    symbol_to_gene_index,
)
from oncorepurpose.evaluation.splits import make_split
from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn_joint
from oncorepurpose.interpret.mechanism_paths import (
    build_mech_index, mechanism_paths, mechanism_score,
)
from oncorepurpose.interpret.paths import _known_pairs
from oncorepurpose.models import HeteroGNN

GENE = "gene_protein"
TARGET = (DRUG_TYPE, "indication", DISEASE_TYPE)
DP = (DRUG_TYPE, "drug_protein", GENE)

# --------------------------------------------------------------------------- #
# Reuse the building blocks of the existing experiments WITHOUT importing them
# as package modules (each guards real work under __main__).
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


emr = _load("_emr_helpers", "evaluate_mechanism_recovery.py")
ef = _load("_ef_helpers", "evaluate_edge_faithfulness.py")
hn = _load("_hn_helpers", "evaluate_hard_negatives.py")

positives = emr.positives
blind_base = emr.blind_base
drug2prot_from = emr.drug2prot_from
gnn_scores = emr.gnn_scores
rank_of = ef.rank_of
all_dp_edges = ef.all_dp_edges
sample_random_removal = ef.sample_random_removal


def _mean(xs):
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs else float("nan")


def _auroc(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones_like(pos), np.zeros_like(neg)]
    s = np.r_[pos, neg]
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# GNN-based tests (1 + 2): one trained joint GNN, evaluated on held-out pairs.
# --------------------------------------------------------------------------- #
def train_joint_model(data, idx, dmdb, sym2gidx, seed, device, epochs, lam):
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
    model = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims).to(device)
    model = train_gnn_joint(model, split, device, mech_tr, decoys, lam=lam, epochs=epochs)
    model.eval()
    return {"model": model, "split": split, "mech_te": mech_te, "decoys": decoys,
            "num_genes": num_genes}


def run_test1_test2(data, idx, ctx, seed, device, n_random, max_pairs):
    """Test 1 (target-edge ablation) + Test 2 (wrong-target substitution).

    Both reuse the same trained model + full-graph embeddings, so test 2 (which
    only re-scores the mechanism head with a swapped gene) is essentially free.
    """
    model, split, mech_te, decoys, num_genes = (
        ctx["model"], ctx["split"], ctx["mech_te"], ctx["decoys"], ctx["num_genes"])
    base = split.base
    d2p = drug2prot_from(base)
    dp_pool = all_dp_edges(base)
    rng = np.random.default_rng(2000 + seed)

    pairs = list(mech_te.pairs)
    n_all = len(pairs)
    if max_pairs:
        pairs = pairs[:max_pairs]

    with torch.no_grad():
        z_full = model.encode(base)

    t1_inst = []          # test 1 faithfulness instances
    t2 = {"true": [], "decoy": [], "rejected": [], "drops": []}  # test 2
    n_skipped = 0
    for (di, ci, genes) in pairs:
        targets = d2p.get(di, set())
        moa = sorted(set(genes) & targets)          # bridge genes that ARE edges
        if not moa:
            n_skipped += 1
            continue
        non_moa = sorted(targets - set(moa))
        n_rm = len(moa)

        s_full = gnn_scores(model, z_full, di, ci, num_genes, device)

        # ----- TEST 1: REMOVE-MOA vs REMOVE-RANDOM -----
        with torch.no_grad():
            z_moa = model.encode(blind_base(base, {(di, g) for g in moa}))
        s_moa = gnn_scores(model, z_moa, di, ci, num_genes, device)

        rand_drops = defaultdict(list)
        rand_rank = defaultdict(list)
        for _ in range(n_random):
            rm, _fb = sample_random_removal(di, non_moa, n_rm, dp_pool, set(moa), rng)
            with torch.no_grad():
                z_r = model.encode(blind_base(base, rm))
            s_r = gnn_scores(model, z_r, di, ci, num_genes, device)
            for g in moa:
                rand_drops[g].append(float(s_full[g] - s_r[g]))
                rand_rank[g].append(rank_of(s_r, g) - rank_of(s_full, g))

        for g in moa:
            rk_full = rank_of(s_full, g)
            t1_inst.append({
                "drug": int(di), "dis": int(ci), "gene": int(g),
                "moa_score_drop": float(s_full[g] - s_moa[g]),
                "rand_score_drop": float(np.mean(rand_drops[g])),
                "moa_rank_degrade": rank_of(s_moa, g) - rk_full,
                "rand_rank_degrade": float(np.mean(rand_rank[g])),
            })

        # ----- TEST 2: wrong-target substitution -----
        # Decoy must be a plausible-but-wrong target: drug-degree-matched, NOT a
        # disease protein, NOT one of this drug's targets, NOT a true bridge gene.
        dis_prots = idx["dis2prot"].get(ci, set())
        exclude = set(genes) | set(targets) | set(dis_prots)
        for g in moa:
            decoy = decoys.sample(int(g), exclude | {int(g)}, 1)
            if not decoy:
                continue
            dg = int(decoy[0])
            true_score = float(s_full[g])
            decoy_score = float(s_full[dg])
            t2["true"].append(true_score)
            t2["decoy"].append(decoy_score)
            t2["rejected"].append(1.0 if true_score > decoy_score else 0.0)
            t2["drops"].append(true_score - decoy_score)

    return {
        "n_test_pairs": n_all, "n_pairs_evaluated": len(pairs),
        "n_pairs_no_moa_edge": n_skipped,
        "t1_instances": t1_inst, "t2": t2,
    }


def summarize_test1(all_inst):
    moa = np.array([r["moa_score_drop"] for r in all_inst], dtype=float)
    rand = np.array([r["rand_score_drop"] for r in all_inst], dtype=float)
    n = len(all_inst)
    frac_faithful = float(np.mean(moa > rand)) if n else float("nan")
    pval = None
    diff = moa - rand
    if n >= 1 and np.any(diff != 0):
        try:
            res = wilcoxon(moa, rand, alternative="greater", zero_method="wilcox")
            pval = float(res.pvalue)
        except Exception as e:  # pragma: no cover
            pval = f"wilcoxon_failed: {e}"
    auroc = _auroc(moa, rand) if n else float("nan")
    return {
        "n_instances": n,
        "mean_moa_score_drop": _mean(moa.tolist()),
        "mean_rand_score_drop": _mean(rand.tolist()),
        "mean_score_contrast": _mean(diff.tolist()),
        "fraction_faithful": frac_faithful,
        "wilcoxon_p_greater": pval,
        "separation_auroc": auroc,
        "mean_moa_rank_degrade": _mean([r["moa_rank_degrade"] for r in all_inst]),
        "mean_rand_rank_degrade": _mean([r["rand_rank_degrade"] for r in all_inst]),
    }


def summarize_test2(t2):
    n = len(t2["true"])
    return {
        "n_instances": n,
        "rejection_rate": _mean(t2["rejected"]),
        "true_vs_decoy_auroc": _auroc(t2["true"], t2["decoy"]),
        "mean_true_score": _mean(t2["true"]),
        "mean_decoy_score": _mean(t2["decoy"]),
        "mean_score_margin": _mean(t2["drops"]),
    }


# --------------------------------------------------------------------------- #
# Test 3: decoy-path swap + literature verifier (graph-extractable MOA path).
# --------------------------------------------------------------------------- #
def run_test3(data, idx, decoys, n_verify, seed, use_llm=False, n_lit=4):
    from oncorepurpose.agent.verify import verify_mechanism

    drug_names = list(data[DRUG_TYPE].node_names)
    dis_names = list(data[DISEASE_TYPE].node_names)
    gene_names = list(data[GENE].node_names)

    onco = hn.oncology_disease_indices(data)
    known = _known_pairs(data)
    ei = data[TARGET].edge_index
    true_pairs = [(int(dr), int(ds)) for dr, ds in zip(ei[0].tolist(), ei[1].tolist())
                  if int(ds) in onco]
    rng = random.Random(seed)
    rng.shuffle(true_pairs)

    records = []
    true_grades, decoy_grades = [], []
    for (di, ci) in true_pairs:
        if len(records) >= n_verify:
            break
        # Real, graph-extractable MOA path (prefer the most specific direct-target).
        paths = mechanism_paths(data, idx, di, ci, max_paths=6)
        if not paths:
            continue
        true_path = paths[0]
        true_genes = [g for g in true_path.get("genes", []) if g]
        if not true_genes:
            continue
        true_sym = true_genes[0]
        # Gene index for the true bridge (seeds the drug-degree-matched decoy).
        true_gidx = _SYM2GIDX.get(true_sym.upper())
        if true_gidx is None:
            continue
        dis_prots = idx["dis2prot"].get(ci, set())
        targets = idx["drug2prot"].get(di, set())
        exclude = set(targets) | set(dis_prots) | {true_gidx}
        decoy = decoys.sample(int(true_gidx), exclude, 1)
        if not decoy:
            continue
        decoy_sym = str(gene_names[int(decoy[0])])

        drug_n = str(drug_names[di])
        dis_n = str(dis_names[ci])
        # Build a matched decoy MOA path: same drug + cancer, wrong bridge gene.
        decoy_path = {
            "type": "direct_target", "len": 2,
            "drug": drug_n, "disease": dis_n, "genes": [decoy_sym], "pathway": None,
            "text": f"{drug_n} --targets--> {decoy_sym} <--associated-- {dis_n}",
        }
        # Real path: reuse the extractor's text but normalize keys for the verifier.
        real_path = {
            "type": true_path.get("type"), "len": true_path.get("len"),
            "drug": drug_n, "disease": dis_n, "genes": true_genes,
            "pathway": true_path.get("pathway"), "text": true_path["text"],
        }
        try:
            v_true = verify_mechanism(real_path, n_lit=n_lit, use_llm=use_llm)
            v_decoy = verify_mechanism(decoy_path, n_lit=n_lit, use_llm=use_llm)
        except Exception as e:  # network hiccup: report + continue
            print(f"  [test3] verify failed for {drug_n}->{dis_n}: {e}")
            continue
        true_grades.append(v_true["grade"])
        decoy_grades.append(v_decoy["grade"])
        records.append({
            "drug": drug_n, "disease": dis_n,
            "true_gene": true_sym, "decoy_gene": decoy_sym,
            "true_grade": v_true["grade"], "decoy_grade": v_decoy["grade"],
            "true_n_abstracts": v_true["n_abstracts"],
            "decoy_n_abstracts": v_decoy["n_abstracts"],
            "source": v_true.get("source"),
        })

    def rate(grades, *labels):
        if not grades:
            return float("nan")
        return float(np.mean([1.0 if g in labels else 0.0 for g in grades]))

    summary = {
        "n_pairs": len(records),
        "use_llm": use_llm,
        "true_grade_dist": dict(Counter(true_grades)),
        "decoy_grade_dist": dict(Counter(decoy_grades)),
        "supported_rate_true": rate(true_grades, "supported"),
        "supported_rate_decoy": rate(decoy_grades, "supported"),
        "supported_or_weak_rate_true": rate(true_grades, "supported", "weak"),
        "supported_or_weak_rate_decoy": rate(decoy_grades, "supported", "weak"),
        "records": records,
    }
    summary["supported_separation"] = (
        summary["supported_rate_true"] - summary["supported_rate_decoy"]
        if summary["supported_rate_true"] == summary["supported_rate_true"] else float("nan"))
    summary["supported_or_weak_separation"] = (
        summary["supported_or_weak_rate_true"] - summary["supported_or_weak_rate_decoy"]
        if summary["supported_or_weak_rate_true"] == summary["supported_or_weak_rate_true"]
        else float("nan"))
    return summary


# --------------------------------------------------------------------------- #
# Test 4: true MOA vs PLAUSIBLE hard negatives (graph mechanism score).
# --------------------------------------------------------------------------- #
def run_test4(data, idx, n_true, n_neg, seed):
    # Mirror evaluate_hard_negatives but with configurable sample sizes.
    hn.N_TRUE, hn.N_NEG = n_true, n_neg
    onco_set = hn.oncology_disease_indices(data)
    onco_list = sorted(onco_set)
    known = _known_pairs(data)
    num_drugs = int(data[DRUG_TYPE].num_nodes)
    num_dis = int(data[DISEASE_TYPE].num_nodes)

    ei = data[TARGET].edge_index
    ind_drug, ind_dis = ei[0].tolist(), ei[1].tolist()
    rng = random.Random(seed)
    true_pairs = [(dr, ds) for dr, ds in zip(ind_drug, ind_dis) if ds in onco_set]
    rng.shuffle(true_pairs)
    true_pairs = true_pairs[:n_true]
    onco_drugs = {dr for dr, ds in zip(ind_drug, ind_dis) if ds in onco_set}

    drug_deg = hn.node_degrees(data, DRUG_TYPE, num_drugs)
    dis_deg = hn.node_degrees(data, DISEASE_TYPE, num_dis)
    drug_bin, drug_bucket = hn.decile_bins(drug_deg, range(num_drugs))
    dis_bin, dis_bucket = hn.decile_bins(dis_deg, onco_set)

    s_true, _, _, ap_true = hn.pair_scores(data, idx, true_pairs)

    negsets = {
        "random": hn.sample_random(random.Random(seed + 1), true_pairs, known, num_drugs, onco_list),
        "degree_matched": hn.sample_degree_matched(
            random.Random(seed + 2), true_pairs, known, drug_bin, drug_bucket, dis_bin, dis_bucket),
        "oncology_drug": hn.sample_oncology_drug(
            random.Random(seed + 3), true_pairs, known, onco_drugs, onco_list),
    }
    try:
        st = hn.sample_shared_target(random.Random(seed + 4), true_pairs, known, idx, onco_set)
        if st:
            negsets["shared_target"] = st
    except Exception as exc:
        print(f"  [test4] shared_target skipped: {exc}")

    results = {}
    for name, negs in negsets.items():
        s_neg, _, _, ap_neg = hn.pair_scores(data, idx, negs)
        results[name] = {
            "n_neg": len(negs),
            "auroc": _auroc(s_true, s_neg),
            "neg_mean_score": float(s_neg.mean()) if len(s_neg) else float("nan"),
            "neg_any_path_rate": float(ap_neg.mean()) if len(ap_neg) else float("nan"),
        }
    return {
        "n_true": len(true_pairs),
        "true_mean_score": float(s_true.mean()),
        "true_any_path_rate": float(ap_true.mean()),
        "negative_types": results,
    }


# --------------------------------------------------------------------------- #
_SYM2GIDX = {}  # populated in main(); used by run_test3 decoy seeding


def _gnn_tests_on_device(data, idx, dmdb, sym2gidx, seeds, device, epochs, lam,
                         n_random, max_pairs):
    """Train + run tests 1 & 2 for each seed on the given device."""
    per_seed = []
    t1_all, t2_pool = [], {"true": [], "decoy": [], "rejected": [], "drops": []}
    decoys_for_t3 = None
    for s in seeds:
        ctx = train_joint_model(data, idx, dmdb, sym2gidx, s, device, epochs, lam)
        if ctx is None:
            print(f"seed {s}: no covered held-out pairs; skipping GNN tests")
            continue
        if decoys_for_t3 is None:
            decoys_for_t3 = ctx["decoys"]
        out = run_test1_test2(data, idx, ctx, s, device, n_random, max_pairs)
        t1_sum = summarize_test1(out["t1_instances"])
        t2_sum = summarize_test2(out["t2"])
        print(f"\nseed {s}: test_pairs={out['n_test_pairs']} "
              f"evaluated={out['n_pairs_evaluated']} no_moa_edge={out['n_pairs_no_moa_edge']}")
        print(f"  [T1] MOA drop={t1_sum['mean_moa_score_drop']:.3f} "
              f"rand drop={t1_sum['mean_rand_score_drop']:.3f} "
              f"contrast={t1_sum['mean_score_contrast']:.3f} "
              f"faithful={t1_sum['fraction_faithful']:.3f} "
              f"AUROC={t1_sum['separation_auroc']:.3f}")
        print(f"  [T2] reject={t2_sum['rejection_rate']:.3f} "
              f"true-vs-decoy AUROC={t2_sum['true_vs_decoy_auroc']:.3f} "
              f"(true={t2_sum['mean_true_score']:.3f} decoy={t2_sum['mean_decoy_score']:.3f})")
        per_seed.append({"seed": s, "n_test_pairs": out["n_test_pairs"],
                         "n_pairs_evaluated": out["n_pairs_evaluated"],
                         "test1": t1_sum, "test2": t2_sum})
        t1_all.extend(out["t1_instances"])
        for k in t2_pool:
            t2_pool[k].extend(out["t2"][k])
    return per_seed, t1_all, t2_pool, decoys_for_t3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-random", type=int, default=2, help="random-removal draws per pair (T1)")
    ap.add_argument("--max-pairs", type=int, default=120, help="cap held-out pairs per seed (T1/T2)")
    ap.add_argument("--n-verify", type=int, default=25, help="pairs for the literature verifier (T3)")
    ap.add_argument("--n-true-hardneg", type=int, default=400, help="true pairs for hard-negative test (T4)")
    ap.add_argument("--n-neg-hardneg", type=int, default=400, help="negatives per class (T4)")
    ap.add_argument("--no-verify", action="store_true", help="skip T3 (network)")
    ap.add_argument("--use-llm", action="store_true", help="use LLM verifier in T3 if a key is set")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.n_random, args.max_pairs = [0], 10, 1, 6
        args.n_verify, args.n_true_hardneg, args.n_neg_hardneg = 3, 40, 40

    seed0 = args.seeds[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} | seeds {args.seeds} | epochs {args.epochs}")

    data, _ = load_primekg(with_features=True)
    idx = build_mech_index(data)
    sym2gidx = symbol_to_gene_index(data)
    global _SYM2GIDX
    _SYM2GIDX = sym2gidx

    dmdb = build_drugmechdb_drug_symbols()
    print(f"DrugMechDB drugs mapped: {len(dmdb)}")

    summary = {
        "title": ("OncoEvidence: Counterfactual Mechanism Stress Test "
                  "(mechanism-guided cancer drug repurposing)"),
        "config": {
            "seeds": args.seeds, "epochs": args.epochs, "lam": args.lam,
            "n_random": args.n_random, "max_pairs": args.max_pairs,
            "n_verify": args.n_verify, "n_true_hardneg": args.n_true_hardneg,
            "n_neg_hardneg": args.n_neg_hardneg, "device_requested": str(device),
            "drugmechdb_drugs_mapped": len(dmdb), "smoke": args.smoke,
        },
    }

    # ---- Tests 1 & 2 (GNN). CUDA OOM -> retry on CPU. ---- #
    per_seed, t1_all, t2_pool, decoys = [], [], None, None
    if not dmdb:
        print("DrugMechDB unavailable -> tests 1 & 2 (curated-gene dependent) BLOCKED; "
              "continuing with tests 3 & 4.")
        summary["config"]["t1_t2_blocked"] = "DrugMechDB unavailable"
    else:
        try:
            per_seed, t1_all, t2_pool, decoys = _gnn_tests_on_device(
                data, idx, dmdb, sym2gidx, args.seeds, device, args.epochs,
                args.lam, args.n_random, args.max_pairs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA OOM ({e}); retrying tests 1 & 2 on CPU.")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                summary["config"]["device_actual"] = "cpu (after CUDA OOM)"
                per_seed, t1_all, t2_pool, decoys = _gnn_tests_on_device(
                    data, idx, dmdb, sym2gidx, args.seeds, device, args.epochs,
                    args.lam, args.n_random, args.max_pairs)
            else:
                raise

    if t1_all:
        summary["test1_target_edge_ablation"] = {
            "description": ("Causal importance: remove the curated drug->target MOA "
                            "edge vs a matched random drug_protein edge of the same "
                            "drug; measure the bridge gene's mechanism-score drop."),
            "per_seed": [p["test1"] for p in per_seed],
            "pooled": summarize_test1(t1_all),
        }
    if t2_pool and t2_pool["true"]:
        summary["test2_wrong_target_substitution"] = {
            "description": ("Specificity: replace the true target with a drug-degree-"
                            "matched decoy NOT associated with the disease; the model "
                            "should score the true MOA above the decoy MOA."),
            "per_seed": [p["test2"] for p in per_seed],
            "pooled": summarize_test2(t2_pool),
        }
    summary["gnn_per_seed_meta"] = [
        {k: v for k, v in p.items() if k in ("seed", "n_test_pairs", "n_pairs_evaluated")}
        for p in per_seed]

    # ---- Test 3 (verifier). Needs a decoy sampler; build one if GNN was skipped. ---- #
    if args.no_verify:
        print("\nTest 3 skipped (--no-verify).")
        summary["test3_decoy_path_verifier"] = {"skipped": "--no-verify"}
    else:
        if decoys is None:
            decoys = DegreeMatchedDecoys(idx["prot_drug_deg"], int(data[GENE].num_nodes), seed=seed0)
        print(f"\nTest 3: literature verifier on {args.n_verify} true vs decoy MOA paths "
              f"(lexical{'+LLM' if args.use_llm else ''})...")
        try:
            t3 = run_test3(data, idx, decoys, args.n_verify, seed0, use_llm=args.use_llm)
            t3["description"] = ("Literature support: swap the real bridge gene for a "
                                 "decoy; the verifier should support the true MOA path "
                                 "more than the decoy. Lexical MOA rubric (no LLM key); "
                                 "an LLM judge would strengthen this.")
            summary["test3_decoy_path_verifier"] = t3
            print(f"  [T3] supported true={t3['supported_rate_true']:.3f} "
                  f"decoy={t3['supported_rate_decoy']:.3f} "
                  f"(sep={t3['supported_separation']:.3f}); "
                  f"supported|weak true={t3['supported_or_weak_rate_true']:.3f} "
                  f"decoy={t3['supported_or_weak_rate_decoy']:.3f} "
                  f"(sep={t3['supported_or_weak_separation']:.3f})")
        except Exception as e:
            print(f"  [T3] BLOCKED ({e}); continuing.")
            summary["test3_decoy_path_verifier"] = {"blocked": str(e)}

    # ---- Test 4 (hard negatives). ---- #
    print(f"\nTest 4: mechanism-score separation vs plausible hard negatives "
          f"({args.n_true_hardneg} true)...")
    t4 = run_test4(data, idx, args.n_true_hardneg, args.n_neg_hardneg, seed0)
    t4["description"] = ("Counterfactual specificity vs non-random negatives: the graph "
                         "mechanism score should still separate true MOA pairs from "
                         "plausible (shared-target / oncology-drug / degree-matched) "
                         "negatives, not just random ones.")
    summary["test4_hard_negatives"] = t4
    for name, r in t4["negative_types"].items():
        print(f"  [T4] {name:<16} AUROC={r['auroc']:.3f} "
              f"(neg any-path={r['neg_any_path_rate']:.1%})")

    summary["verdict"] = build_verdict(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "counterfactual_stress_test.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    out_md = os.path.join(RESULTS_DIR, "counterfactual_stress_test.md")
    with open(out_md, "w") as f:
        f.write(build_markdown(summary))

    print("\n" + "=" * 70)
    print("HEADLINE — Counterfactual Mechanism Stress Test")
    print("=" * 70)
    print(summary["verdict"]["headline"])
    for line in summary["verdict"]["per_test"]:
        print("  - " + line)
    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {out_md}")


# --------------------------------------------------------------------------- #
def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def build_verdict(summary):
    per_test, passes = [], {}

    t1 = summary.get("test1_target_edge_ablation", {}).get("pooled")
    if t1:
        pv = t1["wilcoxon_p_greater"]
        ok = (t1["fraction_faithful"] > 0.5 and isinstance(pv, float) and pv < 0.05
              and t1["mean_score_contrast"] > 0)
        passes["test1"] = ok
        per_test.append(
            f"T1 target-edge ablation: contrast={_fmt(t1['mean_score_contrast'])} "
            f"(MOA={_fmt(t1['mean_moa_score_drop'])} vs random={_fmt(t1['mean_rand_score_drop'])}), "
            f"faithful={_fmt(t1['fraction_faithful'])}, Wilcoxon p={_fmt(pv,2) if isinstance(pv,float) else pv}, "
            f"AUROC={_fmt(t1['separation_auroc'])} -> {'PASS' if ok else 'WEAK'}")

    t2 = summary.get("test2_wrong_target_substitution", {}).get("pooled")
    if t2:
        ok = t2["rejection_rate"] > 0.5 and (not (t2["true_vs_decoy_auroc"] == t2["true_vs_decoy_auroc"])
                                             or t2["true_vs_decoy_auroc"] > 0.5)
        passes["test2"] = ok
        per_test.append(
            f"T2 wrong-target substitution: rejection={_fmt(t2['rejection_rate'])}, "
            f"true-vs-decoy AUROC={_fmt(t2['true_vs_decoy_auroc'])}, "
            f"margin={_fmt(t2['mean_score_margin'])} -> {'PASS' if ok else 'WEAK'}")

    t3 = summary.get("test3_decoy_path_verifier", {})
    if t3 and "supported_separation" in t3:
        sep = t3["supported_separation"]
        sw = t3.get("supported_or_weak_separation", float("nan"))
        ok = (isinstance(sep, float) and sep > 0) or (isinstance(sw, float) and sw > 0)
        passes["test3"] = ok
        per_test.append(
            f"T3 decoy-path verifier: supported true={_fmt(t3['supported_rate_true'])} vs "
            f"decoy={_fmt(t3['supported_rate_decoy'])} (sep={_fmt(sep)}); "
            f"supported|weak sep={_fmt(sw)} -> {'PASS' if ok else 'WEAK'}")
    elif t3.get("skipped") or t3.get("blocked"):
        per_test.append(f"T3 decoy-path verifier: {t3.get('skipped') or t3.get('blocked')}")

    t4 = summary.get("test4_hard_negatives", {}).get("negative_types", {})
    if t4:
        rnd = t4.get("random", {}).get("auroc", float("nan"))
        hardest_name, hardest = None, None
        finite = {k: v["auroc"] for k, v in t4.items() if v["auroc"] == v["auroc"]}
        if finite:
            hardest_name = min(finite, key=finite.get)
            hardest = finite[hardest_name]
        ok = isinstance(hardest, float) and hardest > 0.5
        passes["test4"] = ok
        per_test.append(
            f"T4 hard negatives: random AUROC={_fmt(rnd)}, hardest={hardest_name} "
            f"AUROC={_fmt(hardest)} (>0.5 means mechanism signal survives) -> "
            f"{'PASS' if ok else 'WEAK'}")

    n_pass = sum(1 for v in passes.values() if v)
    n_tot = len(passes)
    moves = n_pass >= max(2, n_tot - 1) and passes.get("test1", False) and passes.get("test2", True)
    headline = (
        f"{n_pass}/{n_tot} counterfactual tests pass. "
        + ("VERDICT: MOVES THE NEEDLE — the model demonstrably relies on the true MOA "
           "edge and rejects fake mechanisms on the tests that probe causal importance "
           "and specificity."
           if moves else
           "VERDICT: PARTIAL — some counterfactual axes hold but at least one specificity "
           "axis is not decisive; see per-test notes and caveats."))
    return {"headline": headline, "passes": passes, "per_test": per_test,
            "moves_the_needle": bool(moves)}


def build_markdown(summary):
    c = summary["config"]
    v = summary["verdict"]
    lines = []
    lines.append("# OncoEvidence — Counterfactual Mechanism Stress Test\n")
    lines.append("> **OncoEvidence: A Counterfactual Evidence-Triage Platform for "
                 "Mechanism-Guided Cancer Drug Repurposing.** The contribution is the "
                 "*evaluation*: we test whether the proposed mechanism path actually "
                 "matters — is it causally important, biologically specific, and "
                 "literature-supported? — rather than whether we beat a ranker.\n")
    lines.append(f"**{v['headline']}**\n")
    lines.append(f"- Config: seeds {c['seeds']}, {c['epochs']} epochs, "
                 f"≤{c['max_pairs']} held-out pairs/seed (T1/T2), {c['n_verify']} verifier "
                 f"pairs (T3), {c['n_true_hardneg']} true pairs (T4); device "
                 f"{c.get('device_actual', c['device_requested'])}; "
                 f"DrugMechDB drugs mapped: {c['drugmechdb_drugs_mapped']}.\n")

    t1 = summary.get("test1_target_edge_ablation", {}).get("pooled")
    if t1:
        pv = t1["wilcoxon_p_greater"]
        pv_s = f"{pv:.2e}" if isinstance(pv, float) else str(pv)
        lines.append("## Test 1 — Target-edge ablation (causal importance)\n")
        lines.append("Remove the curated drug→target MOA edge vs a matched random "
                     "`drug_protein` edge of the same drug, then measure the bridge "
                     "gene's mechanism-score drop.\n")
        lines.append("| Condition | Mean mechanism-score drop |")
        lines.append("|---|---|")
        lines.append(f"| REMOVE-MOA | {_fmt(t1['mean_moa_score_drop'])} |")
        lines.append(f"| REMOVE-RANDOM (matched) | {_fmt(t1['mean_rand_score_drop'])} |")
        lines.append(f"| **Contrast (MOA − random)** | **{_fmt(t1['mean_score_contrast'])}** |\n")
        lines.append(f"- Fraction faithful (MOA hurts more than random): "
                     f"**{_fmt(t1['fraction_faithful'])}** over {t1['n_instances']} instances")
        lines.append(f"- Paired Wilcoxon (MOA drop > random drop): **p = {pv_s}**")
        lines.append(f"- Separation AUROC (MOA vs random drops): **{_fmt(t1['separation_auroc'])}**")
        lines.append(f"- Rank degradation: MOA = {_fmt(t1['mean_moa_rank_degrade'],1)} vs "
                     f"random = {_fmt(t1['mean_rand_rank_degrade'],1)} positions\n")

    t2 = summary.get("test2_wrong_target_substitution", {}).get("pooled")
    if t2:
        lines.append("## Test 2 — Wrong-target substitution (specificity)\n")
        lines.append("Replace the true target with a drug-degree-matched decoy gene that "
                     "is NOT associated with the disease, and score the fake mechanism "
                     "through the trained mechanism head.\n")
        lines.append(f"- **Rejection rate** (true MOA scored above decoy): "
                     f"**{_fmt(t2['rejection_rate'])}** over {t2['n_instances']} instances")
        lines.append(f"- **True-vs-decoy AUROC**: **{_fmt(t2['true_vs_decoy_auroc'])}**")
        lines.append(f"- Mean mechanism score: true = {_fmt(t2['mean_true_score'])} vs "
                     f"decoy = {_fmt(t2['mean_decoy_score'])} "
                     f"(margin {_fmt(t2['mean_score_margin'])})\n")

    t3 = summary.get("test3_decoy_path_verifier", {})
    if t3 and "supported_separation" in t3:
        lines.append("## Test 3 — Decoy-path swap + literature verifier (support)\n")
        lines.append("Keep the drug and cancer, swap the real bridge gene for a decoy "
                     "bridge gene, and run the lexical MOA verifier on Europe PMC "
                     "literature for each. (No LLM key here — the lexical/MOA-rubric path "
                     "is used; an LLM judge would strengthen the precision.)\n")
        lines.append(f"- Verified {t3['n_pairs']} (drug, cancer) pairs")
        lines.append(f"- **Supported rate**: true = **{_fmt(t3['supported_rate_true'])}** vs "
                     f"decoy = **{_fmt(t3['supported_rate_decoy'])}** "
                     f"(separation {_fmt(t3['supported_separation'])})")
        lines.append(f"- Supported|weak rate: true = {_fmt(t3['supported_or_weak_rate_true'])} vs "
                     f"decoy = {_fmt(t3['supported_or_weak_rate_decoy'])} "
                     f"(separation {_fmt(t3['supported_or_weak_separation'])})")
        lines.append(f"- True grade distribution: {t3['true_grade_dist']}")
        lines.append(f"- Decoy grade distribution: {t3['decoy_grade_dist']}\n")
    elif t3.get("skipped") or t3.get("blocked"):
        lines.append("## Test 3 — Decoy-path swap + literature verifier\n")
        lines.append(f"Skipped/blocked: {t3.get('skipped') or t3.get('blocked')}\n")

    t4 = summary.get("test4_hard_negatives", {})
    if t4.get("negative_types"):
        lines.append("## Test 4 — True MOA vs plausible hard negatives (specificity)\n")
        lines.append("Graph mechanism-score separation of true MOA pairs from "
                     "progressively harder, *non-random* negatives.\n")
        lines.append("| Negative class | n | Separation AUROC | neg any-path rate |")
        lines.append("|---|---|---|---|")
        for name, r in t4["negative_types"].items():
            lines.append(f"| {name} | {r['n_neg']} | {_fmt(r['auroc'])} | "
                         f"{_fmt(100*r['neg_any_path_rate'],1)}% |")
        lines.append("")

    lines.append("## Honest reading\n")
    for line in v["per_test"]:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("### Caveats a reviewer would raise\n")
    lines.append("- Tests 1–2 are measured only on held-out (cold-disease, oncology) "
                 "pairs whose curated DrugMechDB bridge gene is a real `drug_protein` "
                 "edge — the population where the counterfactual is well-posed, not all "
                 "predictions. Sample sizes are modest.")
    lines.append("- The \"score\" in tests 1–2 is the mechanism-head logit; drops are in "
                 "logit space (monotonic, not probability). Removing an edge perturbs "
                 "both endpoints' embeddings; the random control isolates *this* edge by "
                 "deleting the same drug's other target edges.")
    lines.append("- Test 2 decoys are drug-degree-matched and disease-unassociated, but a "
                 "decoy could still be a genuine (uncurated) partner; rejection is "
                 "therefore a conservative lower bound on specificity.")
    lines.append("- Test 3 uses the lexical co-mention verifier (no LLM key). Lexical "
                 "grading over-calls \"supported\" relative to an LLM judge; the LLM step "
                 "would sharpen the true-vs-decoy gap. It also depends on Europe PMC "
                 "abstract coverage (OA full text is sparse).")
    lines.append("- Test 4's mechanism score is a hand-designed path score (not learned); "
                 "the shared-target negative (AUROC ≈ 0.6) is honestly the regime where "
                 "the graph signal is weakest, because the decoy drug shares the real "
                 "target.")
    lines.append("- All results are hypothesis-generating and not medical advice.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
