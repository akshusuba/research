#!/usr/bin/env python
"""LLM mechanism-verification eval (OncoEvidence Aim 3, full LLM mode).

Runs the citation-grounded LLM verifier over true oncology indications vs random
drug-cancer pairs, and reports the grade distribution per group plus LLM-vs-lexical
agreement. Requires an OpenRouter / OpenAI-compatible key:

    export ONCO_LLM_API_KEY=...           ONCO_LLM_BASE_URL=https://openrouter.ai/api/v1
    export ONCO_LLM_MODEL=openai/gpt-4o-mini
    PYTHONPATH=. python scripts/verify_llm_eval.py
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from oncorepurpose.agent.verify import verify_mechanism
from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.interpret.mechanism_paths import build_mech_index, mechanism_paths
from oncorepurpose.interpret.paths import _known_pairs

N_PER_GROUP = 50
SEED = 0


def sample(data):
    rng = random.Random(SEED)
    store = data[DISEASE_TYPE]
    onco = set(torch.nonzero(store.is_oncology, as_tuple=False).flatten().tolist()) \
        if "is_oncology" in store else set(range(int(store.num_nodes)))
    known = _known_pairs(data)
    ei = data[(DRUG_TYPE, "indication", DISEASE_TYPE)].edge_index
    true_pairs = [(a, b) for a, b in zip(ei[0].tolist(), ei[1].tolist()) if b in onco]
    rng.shuffle(true_pairs)
    num_drugs = int(data[DRUG_TYPE].num_nodes)
    onco_list = list(onco)
    neg, seen = [], set()
    while len(neg) < N_PER_GROUP * 3 and len(seen) < N_PER_GROUP * 60:
        a, b = rng.randrange(num_drugs), rng.choice(onco_list)
        if (a, b) in known or (a, b) in seen:
            continue
        seen.add((a, b)); neg.append((a, b))
    return true_pairs[:N_PER_GROUP], neg[:N_PER_GROUP]


def run_group(data, idx, pairs, rxnames, dnames):
    rows = []
    for dr, ds in pairs:
        paths = mechanism_paths(data, idx, dr, ds, max_paths=3)
        if not paths:
            rows.append({"drug": rxnames[dr], "disease": dnames[ds],
                         "llm": "no-path", "lexical": "no-path", "path": None})
            continue
        v = verify_mechanism(paths[0], n_lit=4, use_llm=True)
        rows.append({
            "drug": rxnames[dr], "disease": dnames[ds], "path": v["path"],
            "type": v["type"], "llm": (v["llm"] or {}).get("grade", "n/a"),
            "lexical": v["lexical"]["grade"], "source": v["source"],
            "evidence": (v["llm"] or {}).get("evidence", ""),
        })
    return rows


def dist(rows, key):
    return dict(Counter(r[key] for r in rows))


def main():
    if not os.environ.get("ONCO_LLM_API_KEY"):
        print("ONCO_LLM_API_KEY not set; this script needs an LLM key.")
        return
    data, _ = load_primekg(with_features=False)
    idx = build_mech_index(data)
    rxnames = list(data[DRUG_TYPE].node_names)
    dnames = list(data[DISEASE_TYPE].node_names)
    true_pairs, neg_pairs = sample(data)
    print(f"verifying {len(true_pairs)} true + {len(neg_pairs)} random pairs with LLM...")

    true_rows = run_group(data, idx, true_pairs, rxnames, dnames)
    neg_rows = run_group(data, idx, neg_pairs, rxnames, dnames)

    print("\nLLM grade distribution:")
    print("  true  :", dist(true_rows, "llm"))
    print("  random:", dist(neg_rows, "llm"))
    print("\nLexical grade distribution:")
    print("  true  :", dist(true_rows, "lexical"))
    print("  random:", dist(neg_rows, "lexical"))

    graded = [r for r in true_rows + neg_rows if r["path"] and r["llm"] not in ("no-path", "n/a")]
    agree = sum(1 for r in graded if r["llm"] == r["lexical"])
    agreement = agree / len(graded) if graded else None
    print(f"\nLLM-vs-lexical agreement (graded paths, n={len(graded)}): {agreement}")

    supported_true = sum(1 for r in true_rows if r["llm"] == "supported")
    supported_rand = sum(1 for r in neg_rows if r["llm"] == "supported")
    print(f"LLM 'supported': true {supported_true}/{len(true_rows)} | "
          f"random {supported_rand}/{len(neg_rows)}")

    print("\nExample LLM-supported true indications:")
    for r in true_rows:
        if r["llm"] == "supported":
            print(f"   {r['drug']} -> {r['disease']}: {r['evidence'][:110]}")
            if sum(1 for x in true_rows[:true_rows.index(r) + 1] if x['llm'] == 'supported') >= 4:
                break

    out = {
        "model": os.environ.get("ONCO_LLM_MODEL"),
        "n_per_group": N_PER_GROUP,
        "llm_dist": {"true": dist(true_rows, "llm"), "random": dist(neg_rows, "llm")},
        "lexical_dist": {"true": dist(true_rows, "lexical"), "random": dist(neg_rows, "lexical")},
        "llm_vs_lexical_agreement": agreement,
        "supported_rate": {"true": supported_true / len(true_rows),
                           "random": supported_rand / len(neg_rows)},
        "true_rows": true_rows, "random_rows": neg_rows,
    }
    with open(os.path.join(RESULTS_DIR, "verify_llm_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'verify_llm_eval.json')}")


if __name__ == "__main__":
    main()
