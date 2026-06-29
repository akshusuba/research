#!/usr/bin/env python
"""Mechanism-novelty triage for the OncoEvidence repurposing shortlist.

OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided
Cancer Drug Repurposing.

A standing criticism of knowledge-graph repurposing is that the "top hits" are
just rediscovered textbook facts (a known drug for its known target). To answer
that honestly we score *how novel* each shortlisted candidate's mechanism is,
sorting every (drug, cancer) pair into exactly one of four buckets:

  known_mechanism            the drug -> gene MOA we extract is curated in
                             DrugMechDB (this IS the textbook fact).
  known_drug_new_cancer      the drug is already an indicated oncology drug
                             (for a *different* cancer) and there is a plausible
                             MOA path here -- a repositioning, not a new MOA.
  new_mechanism              a specific MOA path exists (direct target / non-hub
                             PPI / pathway) but the drug -> gene MOA is NOT in
                             DrugMechDB -- a genuinely non-obvious hypothesis.
  unsupported_or_hub_artifact  no mechanistic path, or the only paths bridge
                             through a promiscuous hub (CYP / transporter /
                             carrier) and carry little mechanistic signal.

We reuse the project's mechanism-path extractor and the same DrugMechDB
UniProt->HGNC mapping that `scripts/evaluate_mechanism.py` uses, so "known" means
the same thing here as in the curated-agreement metric.

Run:
    PYTHONPATH=. .venv/bin/python scripts/mechanism_novelty.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oncorepurpose.config import DATA_DIR, DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.interpret.mechanism_paths import (
    PROT,
    build_mech_index,
    mechanism_paths,
)
from oncorepurpose.interpret.paths import _known_pairs  # noqa: F401 (documents provenance)

SHORTLIST_PATH = RESULTS_DIR / "repurposing_shortlist.json"
DRUGMECHDB_CACHE = DATA_DIR / "drugmechdb_indication_paths.yaml"
DRUGMECHDB_URLS = [
    "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.yaml",
    "https://raw.githubusercontent.com/SuLab/DrugMechDB/master/indication_paths.yaml",
]

# A bridge/target protein bound by this many distinct drugs is a promiscuous hub
# (the empirical CYP/transporter/carrier machinery sits at the >100 tail of the
# drug-degree distribution; p99 ~= 101). Paths whose drug-target side is such a
# hub carry little mechanistic signal and are treated as artifacts.
HUB_DRUG_DEG = 100

NOVELTY_LABELS = (
    "known_mechanism",
    "known_drug_new_cancer",
    "new_mechanism",
    "unsupported_or_hub_artifact",
)


# --------------------------------------------------------------------------- #
# DrugMechDB curated MOA genes (drug name -> {HGNC symbols})                   #
# --------------------------------------------------------------------------- #
def _looks_like_symbol(name: str) -> bool:
    import re
    name = name.strip().upper()
    if not name or " " in name or "/" in name or "," in name:
        return False
    return bool(re.match(r"^[A-Z0-9][A-Z0-9-]{0,9}$", name)) and any(c.isalpha() for c in name)


def _clean_acc(acc: str) -> str:
    acc = str(acc).strip()
    if acc.lower().startswith("uniprot:"):
        acc = acc.split(":", 1)[1]
    return acc.split("-", 1)[0].strip().upper()


def _load_drugmechdb_raw() -> str | None:
    """Read the cached DrugMechDB YAML; fetch + cache it once if missing."""
    if DRUGMECHDB_CACHE.exists():
        try:
            txt = DRUGMECHDB_CACHE.read_text()
            if len(txt) > 1000:
                return txt
        except Exception:
            pass
    try:
        import requests
    except Exception:
        return None
    for u in DRUGMECHDB_URLS:
        try:
            r = requests.get(u, timeout=45)
            if r.ok and len(r.text) > 1000:
                DRUGMECHDB_CACHE.write_text(r.text)
                return r.text
        except Exception:
            continue
    return None


def load_drugmechdb_symbols():
    """drug name (lower) -> set(HGNC symbols), resolving UniProt accessions.

    Mirrors `scripts/evaluate_mechanism.py::_build_drugmechdb_symbol_map` so a
    candidate is "known_mechanism" here under the same definition used by the
    project's curated-agreement metric.
    """
    raw = _load_drugmechdb_raw()
    if raw is None:
        return None, {"available": False, "reason": "DrugMechDB unreachable / no cache"}
    try:
        import yaml
        entries = yaml.safe_load(raw)
    except Exception as exc:
        return None, {"available": False, "reason": f"parse failed: {exc}"}

    from oncorepurpose.interpret.uniprot_map import uniprot_to_symbol

    drug2accs = defaultdict(set)
    drug2names = defaultdict(set)
    all_accs = set()
    for e in entries:
        g = e.get("graph", {})
        drug = str(g.get("drug", "")).strip().lower()
        if not drug:
            continue
        for n in e.get("nodes", []):
            if str(n.get("label", "")).lower() not in ("protein", "gene"):
                continue
            nid = str(n.get("id", "")).strip()
            if nid.lower().startswith("uniprot:"):
                acc = nid.split(":", 1)[1].strip()
                if acc:
                    drug2accs[drug].add(acc)
                    all_accs.add(acc)
            nm = str(n.get("name", "")).strip()
            if nm and _looks_like_symbol(nm):
                drug2names[drug].add(nm.upper())

    acc2sym = uniprot_to_symbol(all_accs) if all_accs else {}
    drug2symbols = defaultdict(set)
    for drug, accs in drug2accs.items():
        for acc in accs:
            sym = acc2sym.get(_clean_acc(acc))
            if sym:
                drug2symbols[drug].add(sym.upper())
    for drug, names in drug2names.items():
        drug2symbols[drug].update(names)

    meta = {
        "available": True,
        "n_db_drugs": len(drug2symbols),
        "n_uniprot_accessions": len(all_accs),
        "n_accessions_mapped": sum(1 for a in all_accs if acc2sym.get(_clean_acc(a))),
    }
    return drug2symbols, meta


# --------------------------------------------------------------------------- #
# Graph context: oncology diseases + each drug's existing oncology indications #
# --------------------------------------------------------------------------- #
def oncology_disease_set(data) -> set:
    store = data[DISEASE_TYPE]
    if "is_oncology" in store:
        import torch
        return set(torch.nonzero(store.is_oncology, as_tuple=False).flatten().tolist())
    return set(range(int(store.num_nodes)))


def drug_oncology_indications(data) -> dict:
    """drug_idx -> {oncology disease_idx that the drug is indicated for}."""
    onco = oncology_disease_set(data)
    et = (DRUG_TYPE, "indication", DISEASE_TYPE)
    out = defaultdict(set)
    if et in set(data.edge_types):
        ei = data[et].edge_index
        for dr, dz in zip(ei[0].tolist(), ei[1].tolist()):
            if dz in onco:
                out[dr].add(dz)
    return out


# --------------------------------------------------------------------------- #
# Classification                                                              #
# --------------------------------------------------------------------------- #
def classify_novelty(
    data, idx, prot_name2idx, drug2symbols, drug_onco_ind, dz_names,
    drug_idx, disease_idx, drug_name, disease_name,
):
    drug_deg = idx["prot_drug_deg"]

    def deg_of(sym: str) -> int:
        return drug_deg.get(prot_name2idx.get(str(sym).upper()), 0)

    paths = mechanism_paths(data, idx, drug_idx, disease_idx, max_paths=8)
    if not paths:
        return {
            "label": "unsupported_or_hub_artifact",
            "justification": "No drug->target->...->cancer MOA path in the graph "
                             "(only phenotype/non-mechanistic links, if any).",
            "best_path": None,
            "drugmechdb_genes": sorted(drug2symbols.get(drug_name.lower(), [])) if drug2symbols else [],
        }

    # A path is "specific" if its drug-target side (genes[0]) is not a promiscuous hub.
    def hub_deg(p):
        return deg_of(p["genes"][0]) if p.get("genes") else 0

    specific = [p for p in paths if hub_deg(p) < HUB_DRUG_DEG]
    if not specific:
        worst = min(paths, key=lambda p: -hub_deg(p))
        return {
            "label": "unsupported_or_hub_artifact",
            "justification": (
                f"Only paths bridge through a promiscuous hub "
                f"({worst['genes'][0]} bound by {hub_deg(worst)} drugs >= {HUB_DRUG_DEG}); "
                "no specific mechanistic chain."),
            "best_path": worst["text"],
            "drugmechdb_genes": sorted(drug2symbols.get(drug_name.lower(), [])) if drug2symbols else [],
        }

    best = specific[0]
    path_genes = {g.upper() for p in specific for g in p.get("genes", [])}
    db_syms = drug2symbols.get(drug_name.lower()) if drug2symbols else None

    if db_syms and (path_genes & db_syms):
        overlap = sorted(path_genes & db_syms)
        # Display a path that actually contains a curated gene, so the shown MOA
        # chain matches the justification.
        match_path = next(
            (p for p in specific if {g.upper() for g in p.get("genes", [])} & db_syms),
            best,
        )
        return {
            "label": "known_mechanism",
            "justification": (
                f"Extracted MOA gene(s) {overlap} are curated drug->target mechanism "
                f"for {drug_name} in DrugMechDB -- this is a known mechanism."),
            "best_path": match_path["text"],
            "drugmechdb_genes": sorted(db_syms),
        }

    other_onco = sorted((drug_onco_ind.get(drug_idx, set())) - {disease_idx})
    if other_onco:
        ex = [dz_names[i] for i in other_onco[:2]]
        return {
            "label": "known_drug_new_cancer",
            "justification": (
                f"{drug_name} is already an indicated oncology drug "
                f"(e.g. {', '.join(ex)}); a plausible MOA path exists for this new "
                f"cancer via {best['genes']}, but the specific drug->gene MOA is not in DrugMechDB."),
            "best_path": best["text"],
            "drugmechdb_genes": sorted(db_syms) if db_syms else [],
        }

    covered = "covered" if db_syms else "absent"
    return {
        "label": "new_mechanism",
        "justification": (
            f"Specific MOA path via {best['genes']} exists, but the drug->target MOA "
            f"is NOT in DrugMechDB (drug {covered} in DrugMechDB) and {drug_name} is "
            "not an established oncology drug -- a non-obvious, hypothesis-generating mechanism."),
        "best_path": best["text"],
        "drugmechdb_genes": sorted(db_syms) if db_syms else [],
    }


# --------------------------------------------------------------------------- #
# Candidate sourcing                                                          #
# --------------------------------------------------------------------------- #
def load_shortlist_candidates():
    """Return [(disease_name, drug_name), ...] from the saved shortlist, or None."""
    if not SHORTLIST_PATH.exists():
        return None
    with open(SHORTLIST_PATH) as f:
        blob = json.load(f)
    out = []
    for entry in blob.get("shortlist", []):
        dz = entry.get("disease")
        for c in entry.get("candidates", []):
            out.append((dz, c.get("drug")))
    return out or None


def fallback_candidates(data, idx, name2dz):
    """Graph-only fallback if no shortlist exists (no trained model required).

    For a few oncology diseases, rank drugs by mechanism score and keep the top
    few with a real MOA path. (The deliverable path uses the saved shortlist;
    this only fires when it is absent.)"""
    from oncorepurpose.interpret.mechanism_paths import mechanism_score
    import random
    onco = list(oncology_disease_set(data))
    rng = random.Random(0)
    rng.shuffle(onco)
    diseases = onco[:4]
    num_drugs = int(data[DRUG_TYPE].num_nodes)
    drug_names = list(data[DRUG_TYPE].node_names)
    dz_names = list(data[DISEASE_TYPE].node_names)
    drug_sample = rng.sample(range(num_drugs), min(400, num_drugs))
    out = []
    for dz in diseases:
        scored = []
        for dr in drug_sample:
            paths = mechanism_paths(data, idx, dr, dz, max_paths=4)
            if paths:
                scored.append((mechanism_score(paths), dr))
        scored.sort(reverse=True)
        for _, dr in scored[:5]:
            out.append((dz_names[dz], drug_names[dr]))
    return out


# --------------------------------------------------------------------------- #
def main():
    data, _ = load_primekg(with_features=False)
    idx = build_mech_index(data)

    drug_names = list(data[DRUG_TYPE].node_names)
    dz_names = list(data[DISEASE_TYPE].node_names)
    prot_names = list(data[PROT].node_names)
    name2dr = {n: i for i, n in enumerate(drug_names)}
    name2dz = {n: i for i, n in enumerate(dz_names)}
    prot_name2idx = {str(n).upper(): i for i, n in enumerate(prot_names)}

    drug2symbols, dmdb_meta = load_drugmechdb_symbols()
    if dmdb_meta.get("available"):
        print(f"DrugMechDB: {dmdb_meta['n_db_drugs']} drugs, "
              f"{dmdb_meta['n_accessions_mapped']}/{dmdb_meta['n_uniprot_accessions']} "
              f"UniProt accessions mapped to HGNC.")
    else:
        print(f"DrugMechDB unavailable ({dmdb_meta.get('reason')}); "
              "'known_mechanism' cannot be assigned, candidates fall back to other labels.")

    drug_onco_ind = drug_oncology_indications(data)

    candidates = load_shortlist_candidates()
    source = "repurposing_shortlist.json"
    if candidates is None:
        print("No shortlist found; generating a small graph-only fallback pool.")
        candidates = fallback_candidates(data, idx, name2dz)
        source = "graph-only fallback (no trained model)"
    print(f"Scoring {len(candidates)} candidates from {source}.\n")

    rows = []
    skipped = []
    for disease_name, drug_name in candidates:
        di = name2dr.get(drug_name)
        zi = name2dz.get(disease_name)
        if di is None or zi is None:
            skipped.append({"drug": drug_name, "disease": disease_name,
                            "reason": "name not found in graph"})
            continue
        res = classify_novelty(
            data, idx, prot_name2idx, drug2symbols, drug_onco_ind, dz_names,
            di, zi, drug_name, disease_name,
        )
        rows.append({"drug": drug_name, "disease": disease_name, **res})

    counts = Counter(r["label"] for r in rows)
    dist = {lab: counts.get(lab, 0) for lab in NOVELTY_LABELS}

    print("Novelty class distribution:")
    n = max(1, len(rows))
    for lab in NOVELTY_LABELS:
        print(f"  {lab:28s} {dist[lab]:3d}  ({dist[lab]/n:5.1%})")
    if skipped:
        print(f"  (skipped {len(skipped)} candidates not found in graph)")

    non_obvious = dist["new_mechanism"] + dist["known_drug_new_cancer"]
    print(f"\nNon-textbook share (new_mechanism + known_drug_new_cancer): "
          f"{non_obvious}/{len(rows)} = {non_obvious/n:.1%}")
    print(f"Rediscovered-fact share (known_mechanism): "
          f"{dist['known_mechanism']}/{len(rows)} = {dist['known_mechanism']/n:.1%}")

    result = {
        "framing": "OncoEvidence: Counterfactual Evidence-Triage for Mechanism-Guided "
                   "Cancer Drug Repurposing -- mechanism-novelty triage.",
        "source": source,
        "hub_drug_degree_threshold": HUB_DRUG_DEG,
        "drugmechdb": dmdb_meta,
        "n_candidates": len(rows),
        "label_definitions": {
            "known_mechanism": "drug->gene MOA curated in DrugMechDB (textbook fact)",
            "known_drug_new_cancer": "established oncology drug, plausible MOA path, new cancer",
            "new_mechanism": "specific MOA path exists but drug->gene MOA not in DrugMechDB",
            "unsupported_or_hub_artifact": "no specific path / only promiscuous-hub bridge",
        },
        "distribution": dist,
        "non_textbook_share": non_obvious / n,
        "rows": rows,
        "skipped": skipped,
    }
    out_json = RESULTS_DIR / "mechanism_novelty.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # Markdown.
    md = [
        "# OncoEvidence -- Mechanism-Novelty Triage",
        "",
        "_OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided "
        "Cancer Drug Repurposing._",
        "",
        "**Goal (i): rebut the \"rediscovered obvious facts\" criticism.** Every shortlist "
        "candidate is sorted into exactly one novelty bucket using the project's MOA-path "
        "extractor and the same DrugMechDB UniProt->HGNC map used for curated agreement. "
        "Hypothesis-generating; not medical advice.",
        "",
        f"- Source: `{source}`",
        f"- Candidates scored: **{len(rows)}**",
        f"- DrugMechDB available: **{dmdb_meta.get('available')}**"
        + (f" ({dmdb_meta.get('n_db_drugs')} drugs)" if dmdb_meta.get("available") else
           f" ({dmdb_meta.get('reason')})"),
        f"- Promiscuous-hub threshold: drug-degree >= **{HUB_DRUG_DEG}**",
        "",
        "## Novelty class distribution",
        "",
        "| Novelty class | Count | Share |",
        "|---|---|---|",
    ]
    for lab in NOVELTY_LABELS:
        md.append(f"| {lab} | {dist[lab]} | {dist[lab]/n:.1%} |")
    md += [
        "",
        f"**Non-textbook share** (new_mechanism + known_drug_new_cancer): "
        f"**{non_obvious}/{len(rows)} = {non_obvious/n:.1%}**. "
        f"Rediscovered-fact share (known_mechanism): {dist['known_mechanism']/n:.1%}.",
        "",
        "## Example rows",
        "",
        "| Drug | Cancer | Novelty | Best MOA path | Why |",
        "|---|---|---|---|---|",
    ]
    # Show up to 3 examples per label for an honest spread.
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    for lab in NOVELTY_LABELS:
        for r in by_label[lab][:3]:
            path = (r["best_path"] or "-").replace("|", "/")
            why = r["justification"].replace("|", "/")
            md.append(f"| {r['drug']} | {r['disease']} | {lab} | {path} | {why} |")
    (RESULTS_DIR / "mechanism_novelty.md").write_text("\n".join(md) + "\n")

    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {RESULTS_DIR / 'mechanism_novelty.md'}")


if __name__ == "__main__":
    main()
