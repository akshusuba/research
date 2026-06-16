#!/usr/bin/env python
"""Demo: multi-hop mechanism paths for oncology drug-disease pairs on PrimeKG.

Shows that the knowledge graph yields traceable drug -> target -> (PPI/pathway)
-> cancer-gene -> cancer chains for true indications, and contrasts them with a
random (likely-negative) pair, which should yield few/no mechanistic paths.

Run:
    PYTHONPATH=. python scripts/mechanism_demo.py
"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE
from oncorepurpose.datasets import load_primekg
from oncorepurpose.interpret.mechanism_paths import (
    build_mech_index, classify_support, mechanism_paths,
)

ONCO_QUERY = r"glioblastoma|breast carcinoma|melanoma|chronic myeloid|prostate cancer|" \
             r"colorectal|ovarian|pancreatic|lung carcinoma|leukemia"


def main():
    data, targets = load_primekg(with_features=False)
    idx = build_mech_index(data)
    tgt = targets["indication"]
    dnames = list(data[DISEASE_TYPE].node_names)
    rxnames = list(data[DRUG_TYPE].node_names)

    ei = data[tgt].edge_index
    d2drugs = defaultdict(list)
    for dr, ds in zip(ei[0].tolist(), ei[1].tolist()):
        d2drugs[ds].append(dr)

    picks = [i for i, n in enumerate(dnames)
             if n and re.search(ONCO_QUERY, str(n), re.I) and d2drugs.get(i)]

    print("=" * 74)
    print("TRUE INDICATIONS  (mechanism paths should be specific and MOA-like)")
    print("=" * 74)
    shown = 0
    for ds in picks:
        dr = d2drugs[ds][0]
        paths = mechanism_paths(data, idx, dr, ds, max_paths=4)
        if not paths:
            continue
        print(f"\n[{rxnames[dr]}  ->  {dnames[ds]}]   ({classify_support(paths)})")
        for p in paths:
            print(f"   ({p['type']}) {p['text']}")
        shown += 1
        if shown >= 6:
            break

    # Contrast: random drug vs a cancer it is not indicated for.
    print("\n" + "=" * 74)
    print("RANDOM PAIRS  (mostly-negative; expect few/no mechanistic paths)")
    print("=" * 74)
    rng = random.Random(0)
    cancers = picks[:]
    n_with, n_total = 0, 0
    for _ in range(8):
        ds = rng.choice(cancers)
        dr = rng.randrange(len(rxnames))
        if dr in d2drugs[ds]:
            continue
        n_total += 1
        paths = mechanism_paths(data, idx, dr, ds, max_paths=3)
        flag = classify_support(paths)
        if paths:
            n_with += 1
        print(f"\n[{rxnames[dr]}  ->  {dnames[ds]}]   ({flag})")
        for p in paths[:2]:
            print(f"   ({p['type']}) {p['text']}")
    print(f"\nrandom pairs with any mechanistic path: {n_with}/{n_total}")


if __name__ == "__main__":
    main()
