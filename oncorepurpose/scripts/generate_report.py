"""Generate the deliverable: a vetted oncology drug-repurposing shortlist.

Pipeline: train the GNN on PrimeKG indication edges -> for selected cancer
diseases, rank novel drug candidates -> extract multi-hop KG rationales ->
retrieve literature + (optional) LLM evidence dossier + LLM-as-judge score ->
write a ranked markdown + JSON shortlist.

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
from oncorepurpose.interpret.paths import extract_paths, predict_candidates_for_diseases
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
    p.add_argument("--gnn-epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--fallback-features", action="store_true")
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "repurposing_shortlist.json"))
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, targets = load_primekg(with_features=True, force_fallback_features=args.fallback_features)
    target = targets["indication"]

    print("Training deployment GNN on indication edges ...")
    set_all_seeds(0)
    split = make_split(data, target, "transductive", seed=0, val_frac=0.1, test_frac=0.0)
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}
    gnn = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims,
                    hidden=args.hidden, num_layers=2, dropout=0.3)
    gnn = train_gnn(gnn, split, dev, epochs=args.gnn_epochs, patience=10)

    disease_idx = select_diseases(data, args.diseases, args.top_n_diseases)
    print(f"Selected diseases: {[data[DISEASE_TYPE].node_names[i] for i in disease_idx]}")

    preds = predict_candidates_for_diseases(gnn, data, target, disease_idx, dev,
                                            top_k=args.top_k, exclude_known=True)

    use_llm = (not args.no_llm) and llm_available()
    print(f"LLM evidence dossiers: {'ON' if use_llm else 'OFF (no API key or --no-llm)'}")

    shortlist = []
    for dz in disease_idx:
        disease_name = data[DISEASE_TYPE].node_names[dz]
        reports = []
        for drug_i, score in preds[dz]:
            drug_name = data[DRUG_TYPE].node_names[drug_i]
            paths = extract_paths(data, drug_i, dz)
            reports.append(build_candidate_report(drug_name, disease_name, score, paths, use_llm=use_llm))
        reports = rank_reports(reports)
        shortlist.append({"disease": disease_name, "candidates": reports})
        print(f"  {disease_name}: {len(reports)} candidates")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"target": list(target), "llm": use_llm, "shortlist": shortlist}, f, indent=2)

    # Markdown summary.
    md = ["# OncoRepurpose-GNN: candidate repurposing shortlist\n"]
    for entry in shortlist:
        md.append(f"## {entry['disease']}\n")
        for r in entry["candidates"]:
            j = r.get("judge") or {}
            jt = f" | judge plausibility={j.get('plausibility')}, evidence={j.get('evidence_strength')}" if j else ""
            md.append(f"### {r['drug']}  (model score {r['model_score']:.3f}{jt})")
            for pth in r["kg_paths"][:4]:
                md.append(f"- KG path: {pth['text']}")
            if r["literature"]:
                md.append(f"- Literature: {len(r['literature'])} refs (e.g. {r['literature'][0]['title'][:90]})")
            if r.get("dossier"):
                md.append("\n" + r["dossier"] + "\n")
            md.append("")
    out_path.with_suffix(".md").write_text("\n".join(md))
    print(f"\nSaved shortlist to {out_path} and {out_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
