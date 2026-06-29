"""Surface the single best *converging-evidence* repurposing candidate.

OncoEvidence's thesis is that the deliverable is not a score but a mechanism that is
causally important, biologically specific, and literature-supported. This script operationalises
that for the *novel* shortlist: it joins, per candidate, the independent evidence layers we built
and ranks by how many of them converge on the same hypothesis.

Layers joined (each computed by a separate experiment, so agreement is non-trivial):
  1. Graph mechanism            -- a real MOA path exists (direct-target / PPI / pathway), with a
                                   mechanism-strength score and disease-specificity lift.
  2. Conformal calibration      -- split-conformal triage accepts the candidate (confidence >= alpha).
  3. Mechanism novelty          -- not a hub artifact and not a textbook MOA (hypothesis-generating).
  4. Oncology driver context    -- the mechanism touches a known cancer driver family for the disease.
  5. Literature (contradiction) -- Europe PMC scan finds no net contradicting evidence.
  6. (optional) Functional genomics -- if DepMap is on disk, the bridge gene is a real dependency
                                   in the matched cancer lineage (cached so the number is reproducible).

Output: results/converging_evidence.{json,md}. Honest by construction -- a candidate that fails a
layer is shown failing it; the lead is the one where independent layers agree.

Not medical advice; hypothesis-generating only.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"

# Map curated shortlist diseases to DepMap OncotreeLineage labels (best-effort, keyword level).
DISEASE_TO_LINEAGE = {
    "glioblastoma": "CNS/Brain",
    "metastatic melanoma": "Skin",
    "prostate cancer": "Prostate",
    "non-small cell lung carcinoma": "Lung",
    "colorectal cancer": "Bowel",
    "ovarian carcinoma": "Ovary/Fallopian Tube",
}

# Novelty labels we consider hypothesis-generating (vs textbook or hub artifact).
NOVELTY_RANK = {
    "new_mechanism": 3,
    "known_drug_new_cancer": 2,
    "known_mechanism": 1,
    "unsupported_or_hub_artifact": 0,
}


def _norm(s: str) -> str:
    return s.lower().replace(" (disease)", "").strip()


def _load(name: str):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def _lineage_for(disease: str) -> str | None:
    d = _norm(disease)
    for key, lin in DISEASE_TO_LINEAGE.items():
        if key in d:
            return lin
    return None


def depmap_lineage_means(genes_by_lineage: dict[str, set[str]]) -> dict[tuple[str, str], dict]:
    """Batch CRISPR (Chronos) gene-effect: mean over `lineage` cell lines per (lineage, gene).

    Reads the (gitignored) raw DepMap CSV once. More negative = stronger dependency.
    Returns {} if the data is unavailable (the functional layer is then simply skipped).
    """
    crispr = DATA / "depmap" / "CRISPRGeneEffect.csv"
    model = DATA / "depmap" / "Model.csv"
    if not (crispr.exists() and model.exists()):
        return {}
    try:
        import pandas as pd  # local import; only needed for this optional layer

        md = pd.read_csv(model, usecols=["ModelID", "OncotreeLineage"])
        lines_by_lineage = {
            lin: set(md.loc[md["OncotreeLineage"] == lin, "ModelID"])
            for lin in genes_by_lineage
        }
        header = pd.read_csv(crispr, nrows=0).columns.tolist()
        sym2col = {}
        for col in header:  # DepMap columns look like "TP53 (7157)"
            sym2col.setdefault(col.split(" (")[0], col)
        all_genes = set().union(*genes_by_lineage.values()) if genes_by_lineage else set()
        cols = [sym2col[g] for g in all_genes if g in sym2col]
        if not cols:
            return {}
        df = pd.read_csv(crispr, usecols=[header[0]] + cols)
        df = df.set_index(header[0])
        out: dict[tuple[str, str], dict] = {}
        for lin, genes in genes_by_lineage.items():
            sub = df[df.index.isin(lines_by_lineage.get(lin, set()))]
            if sub.empty:
                continue
            for g in genes:
                col = sym2col.get(g)
                if col is None:
                    continue
                v = sub[col].dropna()
                if len(v):
                    out[(lin, g)] = {"mean_chronos": round(float(v.mean()), 3), "n_cell_lines": int(len(v))}
        return out
    except Exception:  # defensive: never fail the surfacing on the optional layer
        return {}


def main() -> None:
    calibrated = _load("repurposing_shortlist_calibrated.json")
    novelty = _load("mechanism_novelty.json")
    subtype = _load("subtype_context.json")
    contra = _load("contradiction_detector.json")
    if calibrated is None:
        raise SystemExit("Run conformal_triage.py first (results/repurposing_shortlist_calibrated.json missing).")

    nov_by = {(_norm(r["disease"]), r["drug"].lower()): r for r in (novelty or {}).get("rows", [])}
    sub_by = {(_norm(c["disease"]), c["drug"].lower()): c for c in (subtype or {}).get("candidates", [])}
    con_by = {(_norm(c["cancer"]), c["drug"].lower()): c for c in (contra or {}).get("shortlist_pairs", [])}

    rows = []
    lifts, mechs = [], []
    flat = []
    for grp in calibrated["shortlist"]:
        for cand in grp["candidates"]:
            flat.append(cand)
            lifts.append(cand.get("specificity_lift", 0.0))
            mechs.append(cand.get("mechanism_score", 0.0))

    # Batch functional-genomics (DepMap) over every candidate's top-path bridge genes, so the
    # converging lead is one where independent CRISPR data ALSO agrees -- not just the graph.
    genes_by_lineage: dict[str, set[str]] = {}
    for cand in flat:
        lin = _lineage_for(cand["disease"])
        genes = cand["kg_paths"][0].get("genes", []) if cand.get("kg_paths") else []
        if lin and genes:
            genes_by_lineage.setdefault(lin, set()).update(genes)
    depmap = depmap_lineage_means(genes_by_lineage)
    depmap_available = bool(depmap)
    # Global gene-effect means let us reward lineage-*specific* dependency rather than
    # pan-essential genes (e.g. PCNA), which are strong dependencies everywhere.
    fg_table = DATA / "fg_depmap_dependency_table.json"
    global_mean = json.load(open(fg_table)).get("global_gene_mean", {}) if fg_table.exists() else {}
    lift_mu = statistics.mean(lifts) if lifts else 0.0
    lift_sd = statistics.pstdev(lifts) or 1.0
    mech_mu = statistics.mean(mechs) if mechs else 0.0
    mech_sd = statistics.pstdev(mechs) or 1.0

    for cand in flat:
        key = (_norm(cand["disease"]), cand["drug"].lower())
        nov = nov_by.get(key, {})
        sub = sub_by.get(key, {})
        con = con_by.get(key, {})
        triage = (cand.get("conformal") or {}).get("triage", "abstain")
        confidence = (cand.get("conformal") or {}).get("calibrated_confidence", 0.0)
        nov_label = nov.get("label", "unknown")
        contradicting = int(con.get("contradicting", 0))
        supporting = int(con.get("supporting", 0))
        aligned = bool(sub.get("driver_context_aligned", False))

        # Functional-genomics: is the top-path bridge gene a real dependency in the matched lineage?
        lineage = _lineage_for(cand["disease"])
        genes = cand["kg_paths"][0].get("genes", []) if cand.get("kg_paths") else []
        fg_vals = [depmap[(lineage, g)]["mean_chronos"] for g in genes if (lineage, g) in depmap]
        fg_mean = round(statistics.mean(fg_vals), 3) if fg_vals else None
        fg_n = depmap[(lineage, genes[0])]["n_cell_lines"] if genes and (lineage, genes[0]) in depmap else None

        # Independent layers that must all pass for a "converging" lead.
        layers = {
            "graph_mechanism": cand.get("mechanism_score", 0.0) > 0,
            "calibration_accept": triage == "accept",
            "novelty_ok": NOVELTY_RANK.get(nov_label, 0) >= 2,  # new mechanism or new cancer
            "driver_context": aligned,
            "no_contradiction": contradicting == 0,
        }
        if depmap_available:
            # Independent CRISPR evidence the model never saw: bridge gene is a dependency.
            layers["functional_dependency"] = fg_mean is not None and fg_mean < -0.10
        n_pass = sum(1 for v in layers.values() if v is True)

        # Transparent convergence score (continuous tie-breaker on top of layer count).
        z_lift = (cand.get("specificity_lift", 0.0) - lift_mu) / lift_sd
        z_mech = (cand.get("mechanism_score", 0.0) - mech_mu) / mech_sd
        # Reward dependency that is *specific* to this lineage (more essential here than globally),
        # not pan-essential genes that score strongly in every cell line.
        if fg_mean is not None and genes:
            g_globals = [global_mean[g] for g in genes if g in global_mean]
            g_global = statistics.mean(g_globals) if g_globals else 0.0
            dep_bonus = max(0.0, g_global - fg_mean)  # >0 iff more dependent in this lineage
        else:
            dep_bonus = 0.0
        conv_score = (
            n_pass
            + 0.25 * z_lift
            + 0.25 * z_mech
            + 0.1 * supporting
            + 0.5 * float(confidence)
            + 0.5 * dep_bonus
        )

        best_path = cand["kg_paths"][0]["text"] if cand.get("kg_paths") else nov.get("best_path", "")
        rows.append(
            {
                "drug": cand["drug"],
                "disease": cand["disease"],
                "n_layers_pass": n_pass,
                "convergence_score": round(conv_score, 3),
                "layers": layers,
                "mechanism_support": cand.get("mechanism_support"),
                "mechanism_score": round(cand.get("mechanism_score", 0.0), 3),
                "specificity_lift": round(cand.get("specificity_lift", 0.0), 3),
                "model_score": round(cand.get("model_score", 0.0), 3),
                "conformal_triage": triage,
                "calibrated_confidence": round(float(confidence), 3),
                "novelty_label": nov_label,
                "novelty_justification": nov.get("justification", ""),
                "driver_context_aligned": aligned,
                "touched_drivers": sub.get("touched_drivers", []),
                "context_rationale": sub.get("context_rationale", ""),
                "supporting_sentences": supporting,
                "contradicting_sentences": contradicting,
                "best_path": best_path,
                "bridge_genes": genes,
                "functional_genomics": (
                    {"lineage": lineage, "mean_chronos": fg_mean, "n_cell_lines": fg_n}
                    if fg_mean is not None else None
                ),
                "top_reference": (cand.get("literature") or [{}])[0].get("title", ""),
            }
        )

    rows.sort(key=lambda r: (r["n_layers_pass"], r["convergence_score"]), reverse=True)
    lead = rows[0]

    out = {
        "framing": (
            "OncoEvidence does not just predict repurposing candidates; it tests whether the "
            "proposed mechanism is causally important, biologically specific, and literature-supported. "
            "This file surfaces the novel candidate where the most independent evidence layers converge."
        ),
        "n_candidates": len(rows),
        "layer_definitions": {
            "graph_mechanism": "a real MOA path (direct-target/PPI/pathway) exists in PrimeKG",
            "calibration_accept": "split-conformal triage accepts (confidence >= alpha)",
            "novelty_ok": "mechanism is new_mechanism or known_drug_new_cancer (not hub artifact / not textbook)",
            "driver_context": "mechanism touches a known cancer-driver family for the disease",
            "no_contradiction": "Europe PMC scan finds no net contradicting evidence",
            "functional_dependency": "(optional) bridge gene is a CRISPR dependency in the matched DepMap lineage",
        },
        "lead": lead,
        "ranking": rows,
    }
    (RESULTS / "converging_evidence.json").write_text(json.dumps(out, indent=2))

    # Markdown digest
    md = []
    md.append("# OncoEvidence: the single best converging-evidence candidate\n")
    md.append("> " + out["framing"] + "\n")
    md.append(
        f"\n**Lead: {lead['drug']} -> {lead['disease']}** "
        f"({lead['n_layers_pass']} independent layers converge; convergence score {lead['convergence_score']}).\n"
    )
    md.append(f"- **Mechanism (graph):** `{lead['best_path']}` "
              f"({lead['mechanism_support']}, score {lead['mechanism_score']}, specificity lift +{lead['specificity_lift']}).\n")
    md.append(f"- **Calibration:** conformal triage = **{lead['conformal_triage']}** "
              f"(confidence {lead['calibrated_confidence']}).\n")
    md.append(f"- **Novelty:** {lead['novelty_label']} -- {lead['novelty_justification']}\n")
    if lead.get("driver_context_aligned"):
        md.append(f"- **Oncology driver context:** {lead['context_rationale']}\n")
    md.append(f"- **Literature:** {lead['supporting_sentences']} supporting vs "
              f"{lead['contradicting_sentences']} contradicting sentences (Europe PMC).\n")
    fg = lead.get("functional_genomics")
    if fg and fg.get("mean_chronos") is not None:
        md.append(f"- **Functional genomics (independent of graph & text):** bridge gene(s) mean CRISPR "
                  f"dependency **{fg['mean_chronos']}** in {fg['n_cell_lines']} {fg['lineage']} cell lines "
                  f"(more negative = stronger; DepMap).\n")
    md.append(f"- Top reference: *{lead['top_reference']}*\n")

    md.append("\n## Top 8 by converging evidence\n")
    md.append("| Drug | Disease | Layers | Conv. | Mech (lift) | Triage | Novelty | Drivers | Dep (Chronos) |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows[:8]:
        dep = (r.get("functional_genomics") or {}).get("mean_chronos")
        dep_s = f"{dep}" if dep is not None else "-"
        md.append(
            f"| {r['drug']} | {_norm(r['disease'])} | {r['n_layers_pass']} | {r['convergence_score']} | "
            f"{r['mechanism_score']} (+{r['specificity_lift']}) | {r['conformal_triage']} | "
            f"{r['novelty_label']} | {','.join(r['touched_drivers']) or '-'} | {dep_s} |\n"
        )
    md.append("\n_Each layer is computed by a separate experiment; agreement across independent layers "
              "is the signal. Hypothesis-generating only; not medical advice._\n")
    (RESULTS / "converging_evidence.md").write_text("".join(md))

    print(f"Lead: {lead['drug']} -> {lead['disease']} "
          f"({lead['n_layers_pass']} layers, score {lead['convergence_score']})")
    print("Wrote results/converging_evidence.{json,md}")


if __name__ == "__main__":
    main()
