#!/usr/bin/env python
"""Timestamped, hash-committed forward prediction set (deliverable 2).

This produces a *prospective registration*: the top mechanism-backed NOVEL
drug->cancer predictions, frozen with a UTC timestamp, the current git commit,
and a sha256 fingerprint over the sorted prediction list. The point is
falsifiability -- this file can be checked, unchanged, against FUTURE
ClinicalTrials.gov entries / approvals to see whether the pipeline called them
before they were established.

Pipeline (mirrors the deployment deliverable in scripts/generate_report.py,
minus the LLM dossier):
  1. train a transductive HeteroGNN on PrimeKG indication edges (candidate
     generator),
  2. rank novel drugs per cancer by disease-specific lift (score minus the
     drug's mean score over random diseases) -- novel = no existing indication /
     contraindication / off-label edge (exclude_known=True),
  3. KEEP only candidates for which the knowledge graph yields a real MOA path
     (direct target / PPI / shared pathway -- not a phenotype coincidence),
  4. rank survivors by mechanism strength then lift, take the top-k per cancer,
  5. freeze: UTC timestamp + git commit (read by file, no git invoked) + sha256.

Run:
    PYTHONPATH=. python scripts/register_predictions.py
    PYTHONPATH=. python scripts/register_predictions.py --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.evaluation.splits import make_split
from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn
from oncorepurpose.interpret.mechanism_paths import (
    build_mech_index, classify_support, mechanism_paths, mechanism_score,
)
from oncorepurpose.interpret.paths import predict_candidates_for_diseases
from oncorepurpose.models import HeteroGNN

HIDDEN = 128


# --------------------------------------------------------------------------- #
def read_git_commit(start: Path):
    """Resolve the current commit hash by reading .git files (no git invoked).

    Walks up from `start` to find a `.git` directory (or gitdir-file), reads
    HEAD, and resolves the symbolic ref against loose refs then packed-refs.
    """
    def resolve(gitdir: Path):
        head_file = gitdir / "HEAD"
        if not head_file.exists():
            return None
        head = head_file.read_text().strip()
        if not head.startswith("ref:"):
            return head  # detached HEAD already holds the hash
        ref = head.split(":", 1)[1].strip()
        loose = gitdir / ref
        if loose.exists():
            return loose.read_text().strip()
        packed = gitdir / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("^"):
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
        return None

    p = Path(start).resolve()
    for d in [p, *p.parents]:
        g = d / ".git"
        if g.is_dir():
            return resolve(g)
        if g.is_file():
            txt = g.read_text().strip()
            if txt.startswith("gitdir:"):
                gd = (d / txt.split(":", 1)[1].strip()).resolve()
                return resolve(gd)
    return None


def _is_oom(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def train_with_fallback(data, split, in_dims, epochs, patience, seed):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for attempt_dev in ([dev, torch.device("cpu")] if dev.type == "cuda" else [dev]):
        try:
            set_all_seeds(seed)
            gnn = HeteroGNN(list(data.node_types), list(split.base.edge_types),
                            in_dims, hidden=HIDDEN, num_layers=2, dropout=0.3)
            gnn = train_gnn(gnn, split, attempt_dev, epochs=epochs, patience=patience)
            return gnn, attempt_dev
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc) and attempt_dev.type == "cuda":
                print("  [oom] CUDA OOM -> falling back to CPU")
                torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError("training failed on all devices")


def top_oncology_diseases(data, n: int):
    """Top-n oncology diseases by indication degree (well-connected cancers)."""
    store = data[DISEASE_TYPE]
    onc = store.is_oncology if "is_oncology" in store else None
    et = (DRUG_TYPE, "indication", DISEASE_TYPE)
    deg = torch.zeros(int(store.num_nodes))
    for d in data[et].edge_index[1].tolist():
        deg[d] += 1
    if onc is not None:
        deg[~onc] = -1.0
    return torch.argsort(deg, descending=True)[:n].tolist()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n-diseases", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5, help="predictions kept per cancer")
    ap.add_argument("--pool", type=int, default=60, help="candidate pool per cancer before mechanism filter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    epochs = args.epochs or (12 if args.smoke else 50)
    n_dis = args.n_diseases or (3 if args.smoke else 12)

    print(f"device(cuda)={torch.cuda.is_available()} | epochs={epochs} | "
          f"n_diseases={n_dis} top_k={args.top_k}")
    data, targets = load_primekg(with_features=True)
    target = targets["indication"]
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}

    # Deployment candidate generator: ALL indication edges seed the graph.
    set_all_seeds(args.seed)
    split = make_split(data, target, "transductive", seed=args.seed,
                       val_frac=0.1, test_frac=0.0)
    gnn, used_dev = train_with_fallback(data, split, in_dims, epochs, args.patience, args.seed)
    print(f"trained candidate-generator GNN on {used_dev}")

    disease_idx = top_oncology_diseases(data, n_dis)
    dis_names = list(data[DISEASE_TYPE].node_names)
    drug_names = list(data[DRUG_TYPE].node_names)
    print(f"cancers: {[dis_names[i] for i in disease_idx]}")

    preds = predict_candidates_for_diseases(
        gnn, data, target, disease_idx, used_dev,
        top_k=args.pool, exclude_known=True, rank_by="specificity",
    )

    print("building mechanism index ...")
    mech_idx = build_mech_index(data)

    predictions = []
    for dz in disease_idx:
        dname = dis_names[dz]
        kept = []
        for drug_i, score, lift in preds[dz]:
            paths = mechanism_paths(data, mech_idx, drug_i, dz, max_paths=6)
            if not paths:
                continue  # NOVEL requires a real MOA path
            kept.append((drug_i, score, lift, paths, mechanism_score(paths)))
        kept.sort(key=lambda t: (t[4], t[2]), reverse=True)
        kept = kept[: args.top_k]
        for drug_i, score, lift, paths, mscore in kept:
            predictions.append({
                "drug": drug_names[drug_i],
                "cancer": dname,
                "model_score": round(float(score), 6),
                "specificity_lift": round(float(lift), 6),
                "mechanism_score": round(float(mscore), 6),
                "mechanism_support": classify_support(paths),
                "moa_paths": [
                    {"type": p["type"], "genes": p["genes"], "text": p["text"],
                     "score": round(float(p["score"]), 6)}
                    for p in paths[:3]
                ],
            })
        print(f"  {dname}: {len(kept)} mechanism-backed novel predictions "
              f"(pool {len(preds[dz])})")

    # Deterministic order, then fingerprint.
    predictions.sort(key=lambda r: (r["cancer"], r["drug"]))
    fingerprint_payload = [
        {"drug": r["drug"], "cancer": r["cancer"],
         "moa_top": r["moa_paths"][0]["text"] if r["moa_paths"] else ""}
        for r in predictions
    ]
    canonical = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    git_commit = read_git_commit(Path(__file__).parent)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d")

    record = {
        "registry": "OncoEvidence forward prediction set",
        "description": ("Timestamped, hash-committed NOVEL mechanism-backed drug->cancer "
                        "predictions for prospective checking against future "
                        "ClinicalTrials.gov entries / approvals. Hypothesis-generating; "
                        "not medical advice."),
        "timestamp_utc": now.isoformat(),
        "git_commit": git_commit,
        "prediction_sha256": sha256,
        "n_predictions": len(predictions),
        "mode": "smoke" if args.smoke else "full",
        "device": str(used_dev),
        "config": {"epochs": epochs, "seed": args.seed, "n_diseases": n_dis,
                   "top_k": args.top_k, "pool": args.pool, "hidden": HIDDEN},
        "target_edge_type": list(target),
        "novelty_criteria": ("no existing indication/contraindication/off-label edge "
                             "(exclude_known=True) AND a non-empty graph MOA path "
                             "(direct target / PPI / shared pathway)"),
        "predictions": predictions,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"registered_predictions_{stamp}.json"
    out_md = RESULTS_DIR / f"registered_predictions_{stamp}.md"
    out_json.write_text(json.dumps(record, indent=2))
    write_markdown(out_md, record)

    print(f"\nRegistered {len(predictions)} novel mechanism-backed predictions.")
    print(f"git_commit={git_commit}")
    print(f"prediction_sha256={sha256}")
    print("examples:")
    for r in predictions[:5]:
        moa = r["moa_paths"][0]["text"] if r["moa_paths"] else "(no path)"
        print(f"  {r['drug']} -> {r['cancer']} | {moa}")
    print(f"saved -> {out_json}")
    print(f"saved -> {out_md}")


def write_markdown(path, r):
    lines = [
        "# OncoEvidence forward prediction set (registered)",
        "",
        f"_{r['timestamp_utc']}_",
        "",
        f"- **git commit:** `{r['git_commit']}`",
        f"- **prediction sha256:** `{r['prediction_sha256']}`",
        f"- **predictions:** {r['n_predictions']}  ·  mode: {r['mode']}  ·  device: {r['device']}",
        "",
        "> Timestamped, hash-committed NOVEL drug→cancer predictions for prospective "
        "checking against future ClinicalTrials.gov entries / approvals. "
        f"Novelty: {r['novelty_criteria']}. Hypothesis-generating; not medical advice.",
        "",
        "## Predictions",
        "",
        "| Cancer | Drug | Support | Model score | Lift | Top MOA path |",
        "|---|---|---|---|---|---|",
    ]
    for p in r["predictions"]:
        moa = p["moa_paths"][0]["text"] if p["moa_paths"] else ""
        lines.append(
            f"| {p['cancer']} | {p['drug']} | {p['mechanism_support']} | "
            f"{p['model_score']:.3f} | {p['specificity_lift']:+.3f} | {moa} |"
        )
    lines += [
        "",
        "## How to check this later",
        "",
        "Re-hash the sorted `predictions` (drug, cancer, top MOA path text) and "
        "confirm it matches `prediction_sha256`; the git commit pins the exact code "
        "that produced them. Then query ClinicalTrials.gov / approvals for each "
        "(drug, cancer) pair dated AFTER this timestamp.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
