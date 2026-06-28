"""Generate the OncoEvidence candidate dossier: a mechanism-grounded shortlist.

This is the deployment-facing deliverable, and it follows the project's actual
thesis: a tuned tabular model is a strong *ranker*, so the graph has to earn its
keep by producing a traceable mechanism, and the LLM/retrieval layer checks
whether that mechanism is supported by the literature. The pipeline is therefore
mechanism-first, not score-first:

1. train a GNN on PrimeKG indication edges (candidate generator only),
2. for each cancer, rank novel drugs by *disease-specific lift* (score minus the
   drug's average score over random diseases) to strip out the popularity
   artifact where a few broadly-indicated drugs top every list,
3. keep only candidates for which the knowledge graph yields a real
   mechanism-of-action path (direct target / PPI / shared pathway -- NOT a
   phenotype/symptom coincidence; see oncorepurpose.interpret.mechanism_paths),
4. rank the survivors by mechanism strength, then retrieve literature and an
   optional LLM evidence dossier + LLM-as-judge score,
5. write a ranked markdown + JSON shortlist.

Everything here is hypothesis-generating, not a vetted clinical recommendation.

Usage:
  python scripts/generate_report.py --diseases glioblastoma "pancreatic cancer" --top-k 5
  ONCO_LLM_API_KEY=... python scripts/generate_report.py   # enables LLM dossiers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from oncorepurpose.agent.evidence_report import build_candidate_report, rank_reports
from oncorepurpose.agent.llm import llm_available
from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.evaluation.splits import make_split
from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn
from oncorepurpose.interpret.mechanism_paths import (
    build_mech_index, classify_support, mechanism_paths, mechanism_score,
)
from oncorepurpose.interpret.paths import predict_candidates_for_diseases
from oncorepurpose.models import HeteroGNN


def select_diseases(data, names: List[str], top_n_fallback: int) -> List[int]:
    disease_names = [str(n).lower() for n in data[DISEASE_TYPE].node_names]
    onc = data[DISEASE_TYPE].is_oncology if "is_oncology" in data[DISEASE_TYPE] else None
    # Indication degree per disease (prefer well-connected diseases).
    et = (DRUG_TYPE, "indication", DISEASE_TYPE)
    deg = torch.zeros(int(data[DISEASE_TYPE].num_nodes))
    for d in data[et].edge_index[1].tolist():
        deg[d] += 1

    selected = []
    for q in names:
        ql = q.lower()
        matches = [i for i, n in enumerate(disease_names) if ql in n]
        if not matches:
            print(f"  [select] no disease match for '{q}'")
            continue
        # Prefer oncology-flagged matches with the most indication edges.
        def rank(i):
            return (1 if (onc is not None and bool(onc[i])) else 0, float(deg[i]))
        best = max(matches, key=rank)
        selected.append(best)
    if not selected:
        # Fallback: top oncology diseases by indication degree.
        onc = data[DISEASE_TYPE].is_oncology
        et = (DRUG_TYPE, "indication", DISEASE_TYPE)
        deg = torch.zeros(int(data[DISEASE_TYPE].num_nodes))
        ei = data[et].edge_index
        for d in ei[1].tolist():
            deg[d] += 1
        deg[~onc] = -1
        selected = torch.argsort(deg, descending=True)[:top_n_fallback].tolist()
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diseases", nargs="*", default=[])
    p.add_argument("--top-n-diseases", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--pool", type=int, default=60,
                   help="candidate pool per disease (ranked by lift) before the mechanism filter")
    p.add_argument("--gnn-epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--fallback-features", action="store_true")
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "repurposing_shortlist.json"))
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, targets = load_primekg(with_features=True, force_fallback_features=args.fallback_features)
    target = targets["indication"]

    print("Training candidate-generator GNN on indication edges ...")
    set_all_seeds(0)
    split = make_split(data, target, "transductive", seed=0, val_frac=0.1, test_frac=0.0)
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}
    gnn = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims,
                    hidden=args.hidden, num_layers=2, dropout=0.3)
    gnn = train_gnn(gnn, split, dev, epochs=args.gnn_epochs, patience=10)

    disease_idx = select_diseases(data, args.diseases, args.top_n_diseases)
    print(f"Selected diseases: {[data[DISEASE_TYPE].node_names[i] for i in disease_idx]}")

    # Generate a generous candidate pool ranked by disease-specific lift, not raw
    # (popularity-saturated) score.
    preds = predict_candidates_for_diseases(
        gnn, data, target, disease_idx, dev,
        top_k=args.pool, exclude_known=True, rank_by="specificity",
    )

    print("Building mechanism index ...")
    mech_idx = build_mech_index(data)

    use_llm = (not args.no_llm) and llm_available()
    print(f"LLM evidence dossiers: {'ON' if use_llm else 'OFF (no API key or --no-llm)'}")

    shortlist = []
    for dz in disease_idx:
        disease_name = data[DISEASE_TYPE].node_names[dz]
        kept = []
        for drug_i, score, lift in preds[dz]:
            # Mechanism filter: the graph must produce a real MOA chain, not a
            # phenotype/symptom coincidence. Candidates with no mechanistic path
            # are dropped here -- this is the graph "earning its place".
            paths = mechanism_paths(data, mech_idx, drug_i, dz, max_paths=6)
            if not paths:
                continue
            kept.append((drug_i, score, lift, paths, mechanism_score(paths)))
        # Rank survivors by mechanism strength, then disease-specific lift.
        kept.sort(key=lambda t: (t[4], t[2]), reverse=True)
        kept = kept[: args.top_k]

        reports = []
        for drug_i, score, lift, paths, mscore in kept:
            drug_name = data[DRUG_TYPE].node_names[drug_i]
            rep = build_candidate_report(drug_name, disease_name, score, paths, use_llm=use_llm)
            rep["specificity_lift"] = lift
            rep["mechanism_score"] = mscore
            rep["mechanism_support"] = classify_support(paths)
            reports.append(rep)
        reports = rank_reports(reports)
        shortlist.append({"disease": disease_name, "candidates": reports})
        print(f"  {disease_name}: {len(reports)} mechanism-backed candidates "
              f"(from a pool of {len(preds[dz])})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"target": list(target), "llm": use_llm, "shortlist": shortlist}, f, indent=2)

    # Markdown summary.
    md = ["# OncoEvidence: mechanism-grounded repurposing shortlist\n",
          "_Candidates are ranked by disease-specific lift and kept only if the knowledge "
          "graph yields a mechanism-of-action path (direct target / PPI / shared pathway). "
          "Phenotype/symptom coincidences are excluded. Hypothesis-generating; not medical "
          "advice._\n"]
    for entry in shortlist:
        md.append(f"## {entry['disease']}\n")
        if not entry["candidates"]:
            md.append("_No mechanism-backed novel candidate in the pool._\n")
            continue
        for r in entry["candidates"]:
            j = r.get("judge") or {}
            jt = (f" | judge plausibility={j.get('plausibility')}, evidence={j.get('evidence_strength')}"
                  if j else "")
            lift = r.get("specificity_lift")
            lt = f" | specificity lift {lift:+.3f}" if lift is not None else ""
            support = r.get("mechanism_support", "")
            md.append(f"### {r['drug']}  ({support} | model score {r['model_score']:.3f}{lt}{jt})")
            for pth in r["kg_paths"][:4]:
                md.append(f"- MOA path: {pth['text']}")
            if r["literature"]:
                md.append(f"- Literature: {len(r['literature'])} refs (e.g. {r['literature'][0]['title'][:90]})")
            if r.get("dossier"):
                md.append("\n" + r["dossier"] + "\n")
            md.append("")
    out_path.with_suffix(".md").write_text("\n".join(md))
    print(f"\nSaved shortlist to {out_path} and {out_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
