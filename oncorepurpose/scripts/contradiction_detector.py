#!/usr/bin/env python
"""OncoEvidence contradiction detector -- look for evidence AGAINST a claim.

A clinically serious triage platform must not only find support for "repurpose
DRUG for CANCER"; it must actively search for evidence that the drug fails, loses
efficacy, or meets resistance in that cancer. This script does exactly that, with
no LLM and no API key (lexical cues only):

  1. For a sample of TRUE oncology indications (known drug->cancer pairs) and for
     every candidate in the current shortlist (``results/repurposing_shortlist.json``
     if present), issue contradiction-oriented Europe PMC queries -- e.g.
     '"{drug}" AND "{cancer}" AND resistance', 'ineffective', 'no benefit',
     '"did not improve"', '"failed trial"'.
  2. Classify each retrieved sentence that co-mentions the drug and the cancer as
     supporting / contradicting / neutral via lexical cues (see
     ``scripts/evidence_lit.py``).
  3. Report a per-pair support-vs-contradiction tally and FLAG any shortlist
     candidate with non-trivial contradicting evidence.

True indications act as a calibration set: a useful contradiction signal should be
relatively low for drugs that genuinely work (though resistance literature is
expected even for effective drugs -- see caveats), and the flag should surface
shortlist candidates whose literature skews negative.

All Europe PMC calls are cached to ``data/europepmc_evidence_cache.json`` and rate
limited (~0.3s between live calls). Network-tolerant: cached data is used if the
network is blocked.

Run:
    PYTHONPATH=. .venv/bin/python scripts/contradiction_detector.py --smoke
    PYTHONPATH=. .venv/bin/python scripts/contradiction_detector.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, DATA_DIR, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.interpret.paths import _known_pairs
from scripts.evidence_lit import EPMCCache, contradiction_scan, query_token

CACHE_PATH = DATA_DIR / "europepmc_evidence_cache.json"
SHORTLIST_PATH = RESULTS_DIR / "repurposing_shortlist.json"
SEED = 0

# A shortlist candidate is flagged when it has at least this many contradicting
# sentences AND the contradiction share of signed sentences exceeds the threshold.
FLAG_MIN_CONTRA = 2
FLAG_MIN_FRACTION = 0.34


def oncology_disease_indices(data):
    store = data[DISEASE_TYPE]
    if "is_oncology" in store:
        return set(torch.nonzero(store.is_oncology, as_tuple=False).flatten().tolist())
    return set(range(int(store.num_nodes)))


def sample_true_pairs(data, n_true):
    rng = random.Random(SEED)
    onco = oncology_disease_indices(data)
    tgt = (DRUG_TYPE, "indication", DISEASE_TYPE)
    ei = data[tgt].edge_index
    pairs = [(dr, ds) for dr, ds in zip(ei[0].tolist(), ei[1].tolist()) if ds in onco]
    rng.shuffle(pairs)
    return pairs[:n_true]


def load_shortlist_pairs():
    """Return [{drug, cancer, model_score, specificity_lift}] from the shortlist file."""
    if not SHORTLIST_PATH.exists():
        return []
    try:
        doc = json.loads(SHORTLIST_PATH.read_text())
    except Exception:
        return []
    out, seen = [], set()
    for disease_block in doc.get("shortlist", []):
        for cand in disease_block.get("candidates", []):
            drug = cand.get("drug", "")
            cancer = query_token(cand.get("disease", disease_block.get("disease", "")))
            key = (drug.lower(), cancer.lower())
            if not drug or not cancer or key in seen:
                continue
            seen.add(key)
            out.append({
                "drug": drug, "cancer": cancer,
                "model_score": cand.get("model_score"),
                "specificity_lift": cand.get("specificity_lift"),
            })
    return out


def scan_group(cache, pairs, label, per_query, max_pairs=None):
    """Run contradiction_scan over (drug, cancer) string pairs; return list of result dicts."""
    rows = []
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    for i, (drug, cancer) in enumerate(pairs):
        res = contradiction_scan(cache, drug, cancer, per_query=per_query)
        res["group"] = label
        rows.append(res)
        if (i + 1) % 10 == 0:
            cache.save()
            print(f"  [{label}] {i+1}/{len(pairs)} scanned "
                  f"(cache hits {cache.stats['hits']}, live {cache.stats['live']}, "
                  f"errors {cache.stats['errors']})")
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {}
    with_contra = [r for r in rows if r["contradicting"] > 0]
    flagged = [r for r in rows
               if r["contradicting"] >= FLAG_MIN_CONTRA and r["contra_fraction"] >= FLAG_MIN_FRACTION]
    tot_support = sum(r["supporting"] for r in rows)
    tot_contra = sum(r["contradicting"] for r in rows)
    return {
        "n_pairs": n,
        "pairs_with_any_contradiction": len(with_contra),
        "pairs_flagged": len(flagged),
        "total_supporting_sentences": tot_support,
        "total_contradicting_sentences": tot_contra,
        "mean_contra_fraction": round(sum(r["contra_fraction"] for r in rows) / n, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny sample for a quick check")
    ap.add_argument("--n-true", type=int, default=None, help="number of true pairs to scan")
    ap.add_argument("--per-query", type=int, default=12)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    n_true = args.n_true or (8 if args.smoke else 60)
    suffix = "_smoke" if args.smoke else ""

    data, _ = load_primekg(with_features=False)
    drug_names = list(data[DRUG_TYPE].node_names)
    dis_names = list(data[DISEASE_TYPE].node_names)

    true_idx_pairs = sample_true_pairs(data, n_true)
    true_pairs = [(drug_names[dr], query_token(dis_names[ds])) for dr, ds in true_idx_pairs]

    shortlist = load_shortlist_pairs()
    if args.smoke:
        shortlist = shortlist[:8]
    print(f"true pairs: {len(true_pairs)} | shortlist candidates: {len(shortlist)}")

    cache = EPMCCache(CACHE_PATH, sleep=args.sleep)

    print("\nScanning TRUE indications for contradicting evidence...")
    true_rows = scan_group(cache, true_pairs, "true", args.per_query)

    print("\nScanning SHORTLIST candidates for contradicting evidence...")
    shortlist_pairs = [(c["drug"], c["cancer"]) for c in shortlist]
    sl_rows = scan_group(cache, shortlist_pairs, "shortlist", args.per_query)
    # attach model metadata back onto shortlist rows
    for r, meta in zip(sl_rows, shortlist):
        r["model_score"] = meta.get("model_score")
        r["specificity_lift"] = meta.get("specificity_lift")
    cache.save()

    true_summary = summarize(true_rows)
    sl_summary = summarize(sl_rows)

    flagged = sorted(
        [r for r in sl_rows
         if r["contradicting"] >= FLAG_MIN_CONTRA and r["contra_fraction"] >= FLAG_MIN_FRACTION],
        key=lambda r: (-r["contra_fraction"], -r["contradicting"]),
    )

    network_blocked = (cache.stats["live"] == 0 and cache.stats["hits"] == 0
                       and cache.stats["errors"] > 0)

    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "config": {"n_true": len(true_pairs), "n_shortlist": len(shortlist),
                   "per_query": args.per_query, "sleep": args.sleep,
                   "flag_min_contra": FLAG_MIN_CONTRA, "flag_min_fraction": FLAG_MIN_FRACTION},
        "epmc_stats": cache.stats,
        "network_blocked": network_blocked,
        "true_summary": true_summary,
        "shortlist_summary": sl_summary,
        "flagged_shortlist": [
            {"drug": r["drug"], "cancer": r["cancer"],
             "contradicting": r["contradicting"], "supporting": r["supporting"],
             "contra_fraction": r["contra_fraction"],
             "model_score": r.get("model_score"),
             "example_sentences": [s for s in r["sentences"] if s["label"] == "contradicting"][:3]}
            for r in flagged
        ],
        "true_pairs": true_rows,
        "shortlist_pairs": sl_rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"contradiction_detector{suffix}.json"
    out_md = RESULTS_DIR / f"contradiction_detector{suffix}.md"
    out_json.write_text(json.dumps(result, indent=2))
    write_markdown(out_md, result)

    # ---- headline numbers ------------------------------------------------ #
    print("\n=== Contradiction detector ===")
    if network_blocked:
        print("NETWORK BLOCKED: no live calls and no cache; results are empty. "
              "Re-run with connectivity to populate the cache.")
    print(f"TRUE indications:      {true_summary.get('n_pairs',0)} pairs | "
          f"{true_summary.get('pairs_with_any_contradiction',0)} with >=1 contradicting sentence | "
          f"{true_summary.get('total_contradicting_sentences',0)} contra vs "
          f"{true_summary.get('total_supporting_sentences',0)} support sentences | "
          f"mean contra-fraction {true_summary.get('mean_contra_fraction',0)}")
    print(f"SHORTLIST candidates:  {sl_summary.get('n_pairs',0)} pairs | "
          f"{sl_summary.get('pairs_with_any_contradiction',0)} with >=1 contradicting sentence | "
          f"{sl_summary.get('pairs_flagged',0)} FLAGGED (>= {FLAG_MIN_CONTRA} contra & "
          f">= {int(FLAG_MIN_FRACTION*100)}% of signed sentences)")
    for r in flagged[:8]:
        print(f"  FLAG  {r['drug']} -> {r['cancer']}: "
              f"contra={r['contradicting']} support={r['supporting']} "
              f"frac={r['contra_fraction']}")
    print(f"\nsaved -> {out_json}\nsaved -> {out_md}")


def write_markdown(path, r):
    ts, t, s = r["timestamp"], r["true_summary"], r["shortlist_summary"]
    lines = [
        f"# Contradiction detector ({r['mode']})",
        "",
        f"_{ts}_",
        "",
        "## What this is",
        "",
        "OncoEvidence's *evidence-against* module. For known indications and for the "
        "current repurposing shortlist, it issues contradiction-oriented Europe PMC "
        "queries (resistance / ineffective / no benefit / did-not-improve / failed-trial) "
        "and grades each drug+cancer co-mention sentence as supporting / contradicting / "
        "neutral using lexical cues (no LLM). This makes the platform look for reasons a "
        "candidate might NOT work, not just reasons it might.",
        "",
    ]
    if r.get("network_blocked"):
        lines += ["> **Network was blocked and no cache was available** -- the tallies "
                  "below are empty. Re-run with connectivity to populate the cache.", ""]
    lines += [
        "## Headline",
        "",
        f"- **True indications ({t.get('n_pairs',0)} pairs):** "
        f"{t.get('pairs_with_any_contradiction',0)} have >=1 contradicting sentence; "
        f"{t.get('total_contradicting_sentences',0)} contradicting vs "
        f"{t.get('total_supporting_sentences',0)} supporting sentences "
        f"(mean contradiction fraction {t.get('mean_contra_fraction',0)}).",
        f"- **Shortlist candidates ({s.get('n_pairs',0)} pairs):** "
        f"{s.get('pairs_with_any_contradiction',0)} have >=1 contradicting sentence; "
        f"**{s.get('pairs_flagged',0)} are FLAGGED** for non-trivial contradicting "
        f"evidence (>= {r['config']['flag_min_contra']} contradicting sentences and "
        f">= {int(r['config']['flag_min_fraction']*100)}% of signed sentences contradicting).",
        "",
        "## Flagged shortlist candidates",
        "",
    ]
    if r["flagged_shortlist"]:
        lines += ["| drug | cancer | contra | support | contra-fraction | model score |",
                  "|---|---|---|---|---|---|"]
        for f in r["flagged_shortlist"]:
            ms = f"{f['model_score']:.3f}" if isinstance(f.get("model_score"), (int, float)) else "-"
            lines.append(f"| {f['drug']} | {f['cancer']} | {f['contradicting']} | "
                         f"{f['supporting']} | {f['contra_fraction']} | {ms} |")
        lines += ["", "### Example contradicting sentences", ""]
        for f in r["flagged_shortlist"][:6]:
            for s_ex in f.get("example_sentences", [])[:1]:
                cues = ", ".join(s_ex.get("cues", [])[:4])
                lines.append(f"- **{f['drug']} / {f['cancer']}** "
                             f"([{s_ex.get('source','')}:{s_ex.get('id','')}], cues: {cues}): "
                             f"\"{s_ex['sentence']}\"")
    else:
        lines.append("_No shortlist candidate crossed the flag threshold._")
    lines += [
        "",
        "## Honest reading & caveats",
        "",
        "- The classifier is **lexical**, not an LLM. 'Resistance' and 'refractory' "
        "appear heavily even for *effective* drugs (mechanism-of-resistance studies, "
        "second-line settings), so a non-zero contradiction tally on a true indication "
        "is expected and does NOT mean the drug fails. The signal is most useful as a "
        "relative flag (which candidates skew negative), not an absolute verdict.",
        "- A negation guard drops cues like 'overcome resistance' / 'no resistance', but "
        "lexical matching still mislabels some sentences (irony, comparison to another "
        "arm, preclinical-vs-clinical).",
        "- Only sentences co-mentioning the drug AND the cancer are counted, which is "
        "conservative (misses pronoun/abbreviation references) but reduces false hits.",
        "- A flag is a prompt for human review, not evidence of inefficacy.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
