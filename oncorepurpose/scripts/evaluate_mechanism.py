#!/usr/bin/env python
"""Evaluate the mechanism-path signal (OncoEvidence Aim 4).

Falsifiable claim: true cancer indications carry stronger graph-mechanistic
structure than random drug-cancer pairs. We test it three ways:

1. Separation (graph-only, no LLM/network): AUROC of the mechanism score
   distinguishing true indications from random (likely-negative) pairs, plus the
   direct-target rate in each group.
2. Literature grounding (small sample, network): run the verifier on a few
   candidates and report the grade distribution for true vs random pairs.
3. DrugMechDB agreement (best-effort): for true pairs whose drug is covered by
   DrugMechDB, check whether our extracted bridge genes overlap the curated MOA
   genes. Skipped gracefully if the resource is unreachable.

Run:
    PYTHONPATH=. python scripts/evaluate_mechanism.py
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, FIGURES_DIR, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.interpret.mechanism_paths import (
    build_mech_index, classify_support, mechanism_paths, mechanism_score,
)
from oncorepurpose.interpret.paths import _known_pairs

N_TRUE = 400
N_NEG = 400
SEED = 0


def oncology_disease_indices(data):
    store = data[DISEASE_TYPE]
    if "is_oncology" in store:
        return set(torch.nonzero(store.is_oncology, as_tuple=False).flatten().tolist())
    return set(range(int(store.num_nodes)))


def sample_pairs(data, idx):
    rng = random.Random(SEED)
    onco = oncology_disease_indices(data)
    known = _known_pairs(data)
    tgt = (DRUG_TYPE, "indication", DISEASE_TYPE)
    ei = data[tgt].edge_index
    true_pairs = [(dr, ds) for dr, ds in zip(ei[0].tolist(), ei[1].tolist()) if ds in onco]
    rng.shuffle(true_pairs)
    true_pairs = true_pairs[:N_TRUE]

    num_drugs = int(data[DRUG_TYPE].num_nodes)
    onco_list = list(onco)
    neg_pairs, seen = [], set()
    while len(neg_pairs) < N_NEG and len(seen) < N_NEG * 20:
        dr = rng.randrange(num_drugs)
        ds = rng.choice(onco_list)
        if (dr, ds) in known or (dr, ds) in seen:
            continue
        seen.add((dr, ds))
        neg_pairs.append((dr, ds))
    return true_pairs, neg_pairs


def score_group(data, idx, pairs):
    scores, direct = [], 0
    for dr, ds in pairs:
        paths = mechanism_paths(data, idx, dr, ds, max_paths=6)
        scores.append(mechanism_score(paths))
        if any(p["type"] == "direct_target" for p in paths):
            direct += 1
    return np.array(scores, dtype=float), direct / max(1, len(pairs))


def grounding_sample(data, idx, true_pairs, neg_pairs, k=6):
    from oncorepurpose.agent.verify import verify_mechanism
    rxnames = list(data[DRUG_TYPE].node_names)
    out = {"true": [], "random": []}
    for label, pairs in [("true", true_pairs[:k]), ("random", neg_pairs[:k])]:
        for dr, ds in pairs:
            paths = mechanism_paths(data, idx, dr, ds, max_paths=3)
            if not paths:
                out[label].append({"drug": rxnames[dr], "grade": "no-path"})
                continue
            v = verify_mechanism(paths[0], n_lit=4, use_llm=True)
            out[label].append({"drug": rxnames[dr], "grade": v["grade"],
                               "source": v["source"], "n_abstracts": v["n_abstracts"]})
    return out


def drugmechdb_agreement(data, idx, true_pairs):
    """Best-effort: overlap of extracted bridge genes with DrugMechDB MOA genes."""
    import requests
    import re as _re
    urls = [
        "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.yaml",
        "https://raw.githubusercontent.com/SuLab/DrugMechDB/master/indication_paths.yaml",
    ]
    raw = None
    for u in urls:
        try:
            r = requests.get(u, timeout=45)
            if r.ok and len(r.text) > 1000:
                raw = r.text
                break
        except Exception:
            continue
    if raw is None:
        return {"available": False, "reason": "DrugMechDB unreachable"}
    try:
        import yaml
        entries = yaml.safe_load(raw)
    except Exception as exc:
        return {"available": False, "reason": f"parse failed: {exc}"}

    # Map drug name -> set of protein/gene node names appearing in its MOA paths.
    drug2genes = defaultdict(set)
    for e in entries:
        g = e.get("graph", {})
        drug = str(g.get("drug", "")).strip().lower()
        if not drug:
            continue
        for n in e.get("nodes", []):
            if str(n.get("label", "")).lower() in ("protein", "gene"):
                nm = str(n.get("name", "")).strip()
                if nm:
                    drug2genes[drug].add(nm.upper())

    rxnames = list(data[DRUG_TYPE].node_names)
    covered, name_overlap = 0, 0
    examples = []
    for dr, ds in true_pairs:
        dn = str(rxnames[dr]).strip().lower()
        if dn not in drug2genes:
            continue
        covered += 1
        paths = mechanism_paths(data, idx, dr, ds, max_paths=8)
        ours = {g.upper() for p in paths for g in p.get("genes", [])}
        overlap = ours & drug2genes[dn]
        if overlap:
            name_overlap += 1
            if len(examples) < 8:
                examples.append({"drug": rxnames[dr], "overlap": sorted(overlap)})
    # DrugMechDB proteins are UniProt accessions / free-text names (e.g. 'BCR/ABL',
    # 'c-Kit', 'topoisomerases II, IV'); PrimeKG uses HGNC symbols (BCR, ABL1, KIT,
    # TOP2A). Direct name overlap is therefore unreliable and understates agreement,
    # so we do NOT report a headline agreement rate -- a UniProt->HGNC map is needed.
    return {"available": True, "n_db_drugs": len(drug2genes),
            "covered_true_pairs": covered,
            "name_overlap_pairs": name_overlap,
            "agreement_rate": None,
            "note": ("DrugMechDB uses UniProt accessions / free-text protein names; "
                     "PrimeKG uses HGNC symbols. Cross-vocabulary mapping (UniProt -> "
                     "HGNC) is required for a meaningful agreement metric; flagged as "
                     "future work and not reported here."),
            "examples": examples}


def main():
    data, _ = load_primekg(with_features=False)
    idx = build_mech_index(data)
    true_pairs, neg_pairs = sample_pairs(data, idx)
    print(f"true pairs: {len(true_pairs)} | random pairs: {len(neg_pairs)}")

    s_true, dt_true = score_group(data, idx, true_pairs)
    s_neg, dt_neg = score_group(data, idx, neg_pairs)
    y = np.r_[np.ones_like(s_true), np.zeros_like(s_neg)]
    s = np.r_[s_true, s_neg]
    auroc = float(roc_auc_score(y, s))
    print(f"\nMechanism-score separation AUROC (true vs random): {auroc:.3f}")
    print(f"  mean score   true={s_true.mean():.3f}  random={s_neg.mean():.3f}")
    print(f"  direct-target rate  true={dt_true:.2%}  random={dt_neg:.2%}")
    print(f"  any-path rate       true={(s_true>0).mean():.2%}  random={(s_neg>0).mean():.2%}")

    print("\nLiterature grounding sample (verifier):")
    ground = grounding_sample(data, idx, true_pairs, neg_pairs)
    for label in ("true", "random"):
        gc = Counter(x["grade"] for x in ground[label])
        print(f"  {label:6s}: {dict(gc)}")

    print("\nDrugMechDB agreement (best-effort):")
    dmdb = drugmechdb_agreement(data, idx, true_pairs)
    if dmdb.get("available"):
        print(f"  covered true pairs: {dmdb['covered_true_pairs']} | "
              f"name-overlap pairs: {dmdb['name_overlap_pairs']} "
              f"(agreement not reported: UniProt<->HGNC mapping needed)")
    else:
        print(f"  skipped: {dmdb.get('reason')}")

    # Figure: score distributions.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(s.max(), 1e-3), 30)
    ax.hist(s_neg, bins=bins, alpha=0.6, label=f"random (mean {s_neg.mean():.2f})", color="#9aa0a6")
    ax.hist(s_true, bins=bins, alpha=0.6, label=f"true indication (mean {s_true.mean():.2f})", color="#e8684a")
    ax.set_xlabel("graph mechanism score")
    ax.set_ylabel("count")
    ax.set_title(f"Mechanism paths separate true vs random oncology pairs (AUROC {auroc:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "mechanism_eval.png"), dpi=150)
    plt.close(fig)

    result = {
        "n_true": len(true_pairs), "n_random": len(neg_pairs),
        "separation_auroc": auroc,
        "mean_score": {"true": float(s_true.mean()), "random": float(s_neg.mean())},
        "direct_target_rate": {"true": dt_true, "random": dt_neg},
        "any_path_rate": {"true": float((s_true > 0).mean()), "random": float((s_neg > 0).mean())},
        "grounding_sample": ground,
        "drugmechdb": dmdb,
    }
    with open(os.path.join(RESULTS_DIR, "mechanism_eval.json"), "w") as f:
        json.dump(result, f, indent=2)

    md = [
        "| Metric | True indications | Random pairs |",
        "|---|---|---|",
        f"| Mean mechanism score | {s_true.mean():.3f} | {s_neg.mean():.3f} |",
        f"| Direct-target rate | {dt_true:.1%} | {dt_neg:.1%} |",
        f"| Any mechanistic path | {(s_true>0).mean():.1%} | {(s_neg>0).mean():.1%} |",
        "",
        f"**Separation AUROC (true vs random): {auroc:.3f}**",
    ]
    with open(os.path.join(RESULTS_DIR, "mechanism_eval.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'mechanism_eval.json')}")


if __name__ == "__main__":
    main()
