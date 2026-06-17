#!/usr/bin/env python
"""Mechanism-recovery experiment (the GNN's home-turf test).

Question: can a GNN trained with an auxiliary mechanism objective name the curated
DrugMechDB bridge gene for a held-out (drug, cancer) pair better than (a) a link-only
GNN, (b) a degree/popularity prior, and (c) a trivial "the drug's own targets"
baseline -- and does any advantage survive removing the direct drug->target edge
(the mechanism-blinded variant)?

We report the answer whichever way it falls. XGBoost has no analogue here: it never
embeds a third (gene) node, so this is an axis where the graph could genuinely add
value -- or honestly fail.

Run:
  PYTHONPATH=. python scripts/evaluate_mechanism_recovery.py --smoke
  PYTHONPATH=. python scripts/evaluate_mechanism_recovery.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE, RESULTS_DIR
from oncorepurpose.datasets import load_primekg
from oncorepurpose.evaluation.mechanism_supervision import (
    DegreeMatchedDecoys, build_drugmechdb_drug_symbols, build_mech_examples,
    symbol_to_gene_index,
)
from oncorepurpose.evaluation.splits import make_split
from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn, train_gnn_joint
from oncorepurpose.interpret.mechanism_paths import build_mech_index
from oncorepurpose.models import HeteroGNN

GENE = "gene_protein"
TARGET = (DRUG_TYPE, "indication", DISEASE_TYPE)
DP = (DRUG_TYPE, "drug_protein", GENE)
PD = (GENE, "drug_protein", DRUG_TYPE)
KS = [5, 10, 20]


def positives(eli, lab):
    return eli[:, lab.bool()]


def rank_metrics(scores: torch.Tensor, true_genes: set):
    order = torch.argsort(scores, descending=True).tolist()
    pos = {g: i for i, g in enumerate(order)}
    best = min((pos[g] for g in true_genes if g in pos), default=None)
    rec = {k: (1.0 if any(pos.get(g, 1e9) < k for g in true_genes) else 0.0) for k in KS}
    mrr = 1.0 / (best + 1) if best is not None else 0.0
    return rec, mrr


@torch.no_grad()
def gnn_scores(model, z, drug_idx, dis_idx, num_genes, device):
    g = torch.arange(num_genes)
    d = torch.full((num_genes,), drug_idx)
    c = torch.full((num_genes,), dis_idx)
    return model.score_mechanism(z, d, g, c).detach().cpu()


@torch.no_grad()
def affinity_scores(z, drug_idx, dis_idx):
    zg = z[GENE].detach()
    s = zg @ z[DRUG_TYPE][drug_idx].detach() + zg @ z[DISEASE_TYPE][dis_idx].detach()
    return s.cpu()


def blind_base(base: HeteroData, remove_pairs: set) -> HeteroData:
    """Copy base, removing drug_protein edges (drug,gene) in remove_pairs (both dirs)."""
    nb = HeteroData()
    for nt in base.node_types:
        for k, v in base[nt].items():
            nb[nt][k] = v
    for et in base.edge_types:
        ei = base[et].edge_index
        if et == DP:
            keep = [j for j in range(ei.size(1)) if (int(ei[0, j]), int(ei[1, j])) not in remove_pairs]
            nb[et].edge_index = ei[:, torch.tensor(keep, dtype=torch.long)] if keep else ei[:, :0]
        elif et == PD:
            keep = [j for j in range(ei.size(1)) if (int(ei[1, j]), int(ei[0, j])) not in remove_pairs]
            nb[et].edge_index = ei[:, torch.tensor(keep, dtype=torch.long)] if keep else ei[:, :0]
        else:
            nb[et].edge_index = ei
    return nb


def drug2prot_from(base: HeteroData):
    d2p = defaultdict(set)
    if DP in base.edge_types:
        ei = base[DP].edge_index
        for a, b in zip(ei[0].tolist(), ei[1].tolist()):
            d2p[a].add(b)
    return d2p


def eval_systems(joint, linkonly, base, test_pairs, idx, num_genes, device, rng):
    """Return per-system aggregated recall@k and MRR over test pairs on `base`."""
    z_joint = joint.encode(base)
    z_link = linkonly.encode(base)
    d2p = drug2prot_from(base)
    deg = np.array([idx["prot_drug_deg"].get(i, 0) for i in range(num_genes)], dtype=float)
    deg_t = torch.tensor(deg)

    agg = {s: {"rec": defaultdict(list), "mrr": []} for s in
           ["joint_gnn", "linkonly_affinity", "target_lookup", "degree_prior"]}
    for di, ci, genes in test_pairs:
        tg = set(genes)
        # joint GNN
        r, m = rank_metrics(gnn_scores(joint, z_joint, di, ci, num_genes, device), tg)
        _acc(agg["joint_gnn"], r, m)
        # link-only embedding affinity
        r, m = rank_metrics(affinity_scores(z_link, di, ci), tg)
        _acc(agg["linkonly_affinity"], r, m)
        # trivial target-lookup: drug's targets first (random order within), others after
        tgt = d2p.get(di, set())
        s = torch.rand(num_genes) * 0.5
        if tgt:
            s[torch.tensor(sorted(tgt), dtype=torch.long)] += 1.0
        r, m = rank_metrics(s, tg)
        _acc(agg["target_lookup"], r, m)
        # degree prior
        r, m = rank_metrics(deg_t + torch.rand(num_genes) * 1e-3, tg)
        _acc(agg["degree_prior"], r, m)
    return {s: {"recall": {k: float(np.mean(agg[s]["rec"][k])) for k in KS},
                "mrr": float(np.mean(agg[s]["mrr"]))} for s in agg}


def _acc(d, rec, mrr):
    for k in KS:
        d["rec"][k].append(rec[k])
    d["mrr"].append(mrr)


def run_seed(data, idx, dmdb, sym2gidx, seed, device, epochs, lam):
    set_all_seeds(seed)
    split = make_split(data, TARGET, "inductive_cold_dst", seed=seed, restrict_oncology=True)
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}
    num_genes = int(data[GENE].num_nodes)

    tr_pos = positives(split.train_label_index, split.train_label)
    te_pos = positives(split.test_label_index, split.test_label)
    mech_tr = build_mech_examples(data, tr_pos, dmdb, sym2gidx)
    mech_te = build_mech_examples(data, te_pos, dmdb, sym2gidx)
    if not mech_te.pairs:
        return None

    decoys = DegreeMatchedDecoys(idx["prot_drug_deg"], num_genes, seed=seed)
    joint = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims).to(device)
    joint = train_gnn_joint(joint, split, device, mech_tr, decoys, lam=lam, epochs=epochs)
    linkonly = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims).to(device)
    linkonly = train_gnn(linkonly, split, device, epochs=epochs)

    rng = np.random.default_rng(seed)
    unblinded = eval_systems(joint, linkonly, split.base, mech_te.pairs, idx, num_genes, device, rng)

    remove = {(di, g) for (di, ci, gs) in mech_te.pairs for g in gs}
    bbase = blind_base(split.base, remove)
    blinded = eval_systems(joint, linkonly, bbase, mech_te.pairs, idx, num_genes, device, rng)
    return {"n_train_pairs": len(mech_tr.pairs), "n_test_pairs": len(mech_te.pairs),
            "unblinded": unblinded, "blinded": blinded}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = [0], 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_primekg(with_features=True)
    idx = build_mech_index(data)
    dmdb = build_drugmechdb_drug_symbols()
    sym2gidx = symbol_to_gene_index(data)
    print(f"DrugMechDB drugs mapped: {len(dmdb)} | device {device}")

    seeds_out = []
    for s in args.seeds:
        r = run_seed(data, idx, dmdb, sym2gidx, s, device, args.epochs, args.lam)
        if r is None:
            print(f"seed {s}: no covered held-out pairs"); continue
        print(f"\nseed {s}: train_pairs={r['n_train_pairs']} test_pairs={r['n_test_pairs']}")
        for cond in ("unblinded", "blinded"):
            print(f"  [{cond}]")
            for sysn, m in r[cond].items():
                print(f"    {sysn:18s} R@5={m['recall'][5]:.3f} R@10={m['recall'][10]:.3f} "
                      f"R@20={m['recall'][20]:.3f} MRR={m['mrr']:.3f}")
        seeds_out.append(r)

    def avg(cond, sysn, field, k=None):
        vals = [(so[cond][sysn]["recall"][k] if k else so[cond][sysn]["mrr"]) for so in seeds_out]
        return float(np.mean(vals)) if vals else None

    summary = {"seeds": args.seeds, "epochs": args.epochs, "lam": args.lam,
               "n_test_pairs": [so["n_test_pairs"] for so in seeds_out],
               "systems": ["joint_gnn", "linkonly_affinity", "target_lookup", "degree_prior"]}
    for cond in ("unblinded", "blinded"):
        summary[cond] = {s: {"recall": {k: avg(cond, s, "rec", k) for k in KS},
                             "mrr": avg(cond, s, "mrr")}
                         for s in summary["systems"]}
    j = summary["unblinded"]["joint_gnn"]["recall"][10]
    t = summary["unblinded"]["target_lookup"]["recall"][10]
    jb = summary["blinded"]["joint_gnn"]["recall"][10]
    tb = summary["blinded"]["target_lookup"]["recall"][10]
    summary["interpretation"] = (
        f"Unblinded R@10: joint_gnn={j}, target_lookup={t}. Blinded R@10: joint_gnn={jb}, "
        f"target_lookup={tb}. The joint GNN adds value only if it beats the trivial "
        f"target-lookup, especially blinded (where the direct drug-target edge is removed).")
    out = os.path.join(RESULTS_DIR, "mechanism_recovery_eval.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + summary["interpretation"])
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
