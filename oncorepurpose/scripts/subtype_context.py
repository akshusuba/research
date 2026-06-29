#!/usr/bin/env python
"""Lightweight cancer subtype / driver-mutation context layer for OncoEvidence.

OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided
Cancer Drug Repurposing.

Goal (ii): make the repurposing rationales *oncology-specific* instead of
generic graph paths. We do this with a small, curated oncogene/driver list,
grouped into mechanistic context families (RTK/EGFR, PI3K/AKT, RAS/MAPK, p53,
cell-cycle/RB, WNT, Notch, DNA-repair, hormone, IDH/metabolic, kinase-fusion).

For each cancer (disease node) we intersect its disease-associated proteins
(`disease_protein` edges) with the driver list to assign a *driver context*
(e.g. glioblastoma -> {EGFR (RTK/EGFR), PTEN (PI3K/AKT), TP53 (p53)}). We then
re-express each shortlisted candidate's rationale in that context, e.g.

    "Tamoxifen targets PRKCA in a PI3K/AKT-driven context of glioblastoma."

and report, for the shortlist, how many candidates have a mechanism path that
*touches* the cancer's driver context -- a crude but meaningful specificity
signal (the path connects to a known driver of that exact cancer, not just any
disease-associated protein).

Run:
    PYTHONPATH=. .venv/bin/python scripts/subtype_context.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.interpret.mechanism_paths import (
    PROT,
    build_mech_index,
    mechanism_paths,
)

SHORTLIST_PATH = RESULTS_DIR / "repurposing_shortlist.json"

# Curated oncogene/driver list, grouped into mechanistic context families. HGNC
# symbols (PrimeKG gene_protein node names are HGNC symbols). Deliberately small
# and canonical -- this is a context layer, not a comprehensive driver atlas.
DRIVER_FAMILIES = {
    "RTK/EGFR": ["EGFR", "ERBB2", "ERBB3", "ERBB4", "MET", "KIT", "PDGFRA",
                 "FGFR1", "FGFR2", "FGFR3", "ALK", "ROS1", "RET", "FLT3", "KDR"],
    "PI3K/AKT/mTOR": ["PIK3CA", "PIK3R1", "PTEN", "AKT1", "AKT2", "MTOR", "TSC1", "TSC2"],
    "RAS/MAPK": ["KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "MAP2K1", "MAPK1", "NF1"],
    "p53": ["TP53", "MDM2", "MDM4"],
    "cell-cycle/RB": ["RB1", "CDKN2A", "CDK4", "CDK6", "CCND1", "CCNE1", "MYC", "MYCN", "E2F1"],
    "WNT": ["CTNNB1", "APC", "AXIN1"],
    "Notch": ["NOTCH1", "NOTCH2", "NOTCH3"],
    "DNA-repair": ["BRCA1", "BRCA2", "ATM", "ATR", "MLH1", "MSH2", "PARP1", "PALB2"],
    "hormone": ["ESR1", "ESR2", "AR", "PGR"],
    "IDH/metabolic": ["IDH1", "IDH2", "VHL", "FH", "SDHB"],
    "kinase-fusion": ["ABL1", "BCR", "JAK2", "JAK1", "PML", "RARA"],
}
SYMBOL2FAMILY = {sym: fam for fam, syms in DRIVER_FAMILIES.items() for sym in syms}
DRIVER_SYMBOLS = set(SYMBOL2FAMILY)


def driver_context_for_disease(data, idx, prot_names, disease_idx):
    """Return (driver_genes_sorted, families_sorted) for a cancer node.

    driver_genes = disease-associated proteins that are curated drivers.
    families     = the mechanistic context families those drivers span.
    """
    dis_prots = idx["dis2prot"].get(disease_idx, set())
    genes = set()
    for p in dis_prots:
        sym = str(prot_names[p]).upper() if p < len(prot_names) else None
        if sym in DRIVER_SYMBOLS:
            genes.add(sym)
    families = sorted({SYMBOL2FAMILY[g] for g in genes})
    return sorted(genes), families


def load_shortlist():
    if not SHORTLIST_PATH.exists():
        return None
    with open(SHORTLIST_PATH) as f:
        return json.load(f)


def context_aware_rationale(drug, disease, path, touched_genes):
    """Re-express a MOA path as a driver-context-aware sentence."""
    target = path["genes"][0] if path.get("genes") else "its target"
    fams = sorted({SYMBOL2FAMILY[g] for g in touched_genes})
    fam_str = "/".join(fams) if fams else "driver"
    via = ""
    if len(path["genes"]) > 1:
        via = f" (reaching {path['genes'][-1]})"
    return (f"{drug} targets {target}{via} in a {fam_str}-driven context of "
            f"{disease} [driver hit: {', '.join(sorted(touched_genes))}].")


def main():
    data, _ = load_primekg(with_features=False)
    idx = build_mech_index(data)

    drug_names = list(data[DRUG_TYPE].node_names)
    dz_names = list(data[DISEASE_TYPE].node_names)
    prot_names = list(data[PROT].node_names)
    name2dr = {n: i for i, n in enumerate(drug_names)}
    name2dz = {n: i for i, n in enumerate(dz_names)}

    blob = load_shortlist()
    if blob is None:
        print("No repurposing_shortlist.json found; nothing to contextualize.")
        # Still emit driver contexts for a few oncology diseases as a demo.
        import torch
        store = data[DISEASE_TYPE]
        onco = (torch.nonzero(store.is_oncology, as_tuple=False).flatten().tolist()
                if "is_oncology" in store else list(range(min(50, int(store.num_nodes)))))
        diseases = [(dz_names[i], i) for i in onco[:6]]
        candidates_by_disease = {dz: [] for dz, _ in diseases}
    else:
        diseases = []
        candidates_by_disease = {}
        for entry in blob.get("shortlist", []):
            dzn = entry.get("disease")
            zi = name2dz.get(dzn)
            if zi is None:
                continue
            diseases.append((dzn, zi))
            candidates_by_disease[dzn] = [c.get("drug") for c in entry.get("candidates", [])]

    # 1) Driver context per cancer.
    disease_contexts = {}
    print("Driver context per cancer (disease_protein  drivers):")
    for dzn, zi in diseases:
        genes, fams = driver_context_for_disease(data, idx, prot_names, zi)
        disease_contexts[dzn] = {"driver_genes": genes, "families": fams}
        print(f"  {dzn:45s} {fams}  {genes}")

    # 2) Candidate-level driver-context alignment.
    aligned = 0
    total = 0
    cand_rows = []
    for dzn, zi in diseases:
        ctx_genes = set(disease_contexts[dzn]["driver_genes"])
        for drug in candidates_by_disease.get(dzn, []):
            di = name2dr.get(drug)
            if di is None:
                continue
            total += 1
            paths = mechanism_paths(data, idx, di, zi, max_paths=8)
            # A path "touches" the driver context if any of its genes is a driver
            # of this exact cancer.
            touch_path = None
            touched = set()
            for p in paths:
                hit = {g.upper() for g in p.get("genes", [])} & ctx_genes
                if hit:
                    touch_path = p
                    touched = hit
                    break
            is_aligned = touch_path is not None
            aligned += int(is_aligned)
            row = {
                "drug": drug,
                "disease": dzn,
                "driver_context_families": disease_contexts[dzn]["families"],
                "driver_context_aligned": is_aligned,
                "touched_drivers": sorted(touched),
                "context_rationale": (
                    context_aware_rationale(drug, dzn, touch_path, touched)
                    if is_aligned else
                    f"{drug} has a MOA path in {dzn} but it does not touch a curated "
                    f"driver of this cancer ({disease_contexts[dzn]['families'] or 'no drivers found'})."),
                "best_path": (touch_path["text"] if touch_path else
                              (paths[0]["text"] if paths else None)),
            }
            cand_rows.append(row)

    rate = aligned / total if total else 0.0
    print(f"\nDriver-context-aligned candidates: {aligned}/{total} = {rate:.1%}")
    print("\nExample context-aware rationales:")
    for r in [x for x in cand_rows if x["driver_context_aligned"]][:6]:
        print(f"  - {r['context_rationale']}")

    result = {
        "framing": "OncoEvidence: Counterfactual Evidence-Triage for Mechanism-Guided "
                   "Cancer Drug Repurposing -- subtype / driver-context layer.",
        "driver_families": DRIVER_FAMILIES,
        "disease_contexts": disease_contexts,
        "n_candidates": total,
        "n_driver_context_aligned": aligned,
        "driver_context_alignment_rate": rate,
        "candidates": cand_rows,
    }
    out_json = RESULTS_DIR / "subtype_context.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    md = [
        "# OncoEvidence -- Subtype / Driver-Context Layer",
        "",
        "_OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided "
        "Cancer Drug Repurposing._",
        "",
        "**Goal (ii): make rationales oncology-specific.** A small curated driver list "
        "(grouped into mechanistic families) is intersected with each cancer's "
        "`disease_protein` neighbours to assign a *driver context*. Candidate rationales "
        "are re-expressed in that context, and we report how many candidates have a MOA "
        "path that touches a curated driver of that exact cancer. Hypothesis-generating; "
        "not medical advice.",
        "",
        "## Driver context per cancer",
        "",
        "| Cancer | Context families | Driver genes (disease_protein ∩ drivers) |",
        "|---|---|---|",
    ]
    for dzn, _ in diseases:
        c = disease_contexts[dzn]
        md.append(f"| {dzn} | {', '.join(c['families']) or '-'} | "
                  f"{', '.join(c['driver_genes']) or '-'} |")
    md += [
        "",
        f"## Candidate driver-context alignment: **{aligned}/{total} = {rate:.1%}**",
        "",
        "| Drug | Cancer | Aligned | Touched driver(s) | Context-aware rationale |",
        "|---|---|---|---|---|",
    ]
    for r in cand_rows:
        rationale = r["context_rationale"].replace("|", "/")
        md.append(f"| {r['drug']} | {r['disease']} | "
                  f"{'yes' if r['driver_context_aligned'] else 'no'} | "
                  f"{', '.join(r['touched_drivers']) or '-'} | {rationale} |")
    (RESULTS_DIR / "subtype_context.md").write_text("\n".join(md) + "\n")

    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {RESULTS_DIR / 'subtype_context.md'}")


if __name__ == "__main__":
    main()
