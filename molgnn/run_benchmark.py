"""Rigorous GNN vs XGBoost head-to-head on EGFR bioactivity.

Same molecules for both models. Primary split: Bemis-Murcko scaffold.
Variation: scaffolds are RE-SPLIT per seed (tie-break shuffle) AND model init
varies per seed. Reports AUROC + AUPRC mean +/- std over >=3 seeds.
"""
import argparse
import json
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

import data_utils as du
from models import GIN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV = "/home/elrarun/code/research/molgnn/data/egfr_raw.csv"


# ----------------- GNN -----------------
GNN_GRID = [
    {"num_layers": 4, "hidden": 128, "pool": "mean", "dropout": 0.3, "lr": 1e-3},
    {"num_layers": 5, "hidden": 128, "pool": "add", "dropout": 0.3, "lr": 1e-3},
    {"num_layers": 3, "hidden": 256, "pool": "mean", "dropout": 0.5, "lr": 5e-4},
]


def _train_gnn_one(tl, vl, cfg, pos_weight, seed, max_epochs=150, patience=20):
    torch.manual_seed(seed)
    model = GIN(du.ATOM_FDIM, hidden=cfg["hidden"], num_layers=cfg["num_layers"],
                dropout=cfg["dropout"], pool=cfg["pool"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auprc, best_state, wait = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for batch in tl:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            crit(out, batch.y).backward()
            opt.step()
        val_auprc, _ = eval_gnn(model, vl)
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val_auprc


def train_gnn(graphs, train_idx, val_idx, test_idx, seed):
    """Small architecture grid selected on val AUPRC (mirrors XGB grid fairness)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = [graphs[i] for i in train_idx]
    va = [graphs[i] for i in val_idx]
    te = [graphs[i] for i in test_idx]

    ytr = np.array([float(g.y.item()) for g in tr])
    pos = ytr.sum(); neg = len(ytr) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=DEVICE)

    tl = DataLoader(tr, batch_size=128, shuffle=True)
    vl = DataLoader(va, batch_size=256)
    tel = DataLoader(te, batch_size=256)

    best_val, best_model, best_cfg = -1, None, None
    for cfg in GNN_GRID:
        model, v = _train_gnn_one(tl, vl, cfg, pos_weight, seed)
        if v > best_val:
            best_val, best_model, best_cfg = v, model, cfg
    test_auprc, test_auroc = eval_gnn(best_model, tel)
    return {"auroc": test_auroc, "auprc": test_auprc, "best_val_auprc": best_val,
            "best_cfg": best_cfg}


@torch.no_grad()
def eval_gnn(model, loader):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch.x, batch.edge_index, batch.batch)
        ps.append(torch.sigmoid(out).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return average_precision_score(y, p), roc_auc_score(y, p)


# ----------------- XGBoost (tuned) -----------------
def train_xgb(fps, labels, train_idx, val_idx, test_idx, seed):
    import xgboost as xgb
    Xtr, Xva, Xte = fps[train_idx], fps[val_idx], fps[test_idx]
    ytr, yva, yte = labels[train_idx], labels[val_idx], labels[test_idx]

    pos = ytr.sum(); neg = len(ytr) - pos
    spw = neg / max(pos, 1)

    grid = [
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 400, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 600, "subsample": 0.8, "colsample_bytree": 0.6},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 400, "subsample": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 600, "subsample": 0.8, "colsample_bytree": 0.5},
        {"max_depth": 10, "learning_rate": 0.03, "n_estimators": 800, "subsample": 0.7, "colsample_bytree": 0.5},
    ]
    best_val, best_model = -1, None
    for params in grid:
        clf = xgb.XGBClassifier(
            **params, scale_pos_weight=spw, tree_method="hist", device="cuda",
            eval_metric="aucpr", random_state=seed, n_jobs=8,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        vp = clf.predict_proba(Xva)[:, 1]
        v = average_precision_score(yva, vp)
        if v > best_val:
            best_val, best_model = v, clf
    tp = best_model.predict_proba(Xte)[:, 1]
    return {"auroc": roc_auc_score(yte, tp), "auprc": average_precision_score(yte, tp),
            "best_val_auprc": best_val}


# ----------------- RandomForest (context) -----------------
def train_rf(fps, labels, train_idx, val_idx, test_idx, seed):
    from sklearn.ensemble import RandomForestClassifier
    Xtr = fps[np.concatenate([train_idx, val_idx])]
    ytr = labels[np.concatenate([train_idx, val_idx])]
    Xte, yte = fps[test_idx], labels[test_idx]
    clf = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=8,
                                 class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    tp = clf.predict_proba(Xte)[:, 1]
    return {"auroc": roc_auc_score(yte, tp), "auprc": average_precision_score(yte, tp)}


def summarize(runs):
    out = {}
    for model in runs[0]:
        for metric in ("auroc", "auprc"):
            vals = [r[model][metric] for r in runs]
            out[f"{model}_{metric}_mean"] = float(np.mean(vals))
            out[f"{model}_{metric}_std"] = float(np.std(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=6.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", choices=["scaffold", "random", "both"], default="both")
    args = ap.parse_args()

    t0 = time.time()
    ds = du.load_dataset(CSV, threshold=args.threshold)
    n = len(ds["labels"])
    print(f"N={n}, active={int(ds['labels'].sum())} "
          f"({ds['labels'].mean()*100:.1f}%), feat_dim={du.ATOM_FDIM}, device={DEVICE}")
    n_scaffolds = len(set(ds["scaffolds"]))
    print(f"Unique Bemis-Murcko scaffolds: {n_scaffolds}")

    results = {"dataset": {
        "target": "EGFR (CHEMBL203), Homo sapiens",
        "source": "ChEMBL bioactivities with pchembl_value (median-aggregated per molecule)",
        "n_molecules": int(n),
        "threshold_active": args.threshold,
        "n_active": int(ds["labels"].sum()),
        "frac_active": float(ds["labels"].mean()),
        "n_scaffolds": int(n_scaffolds),
        "atom_feat_dim": int(du.ATOM_FDIM),
        "fingerprint": "Morgan/ECFP radius=2, 2048 bits",
    }, "splits": {}}

    split_types = ["scaffold", "random"] if args.split == "both" else [args.split]
    for stype in split_types:
        print(f"\n===== SPLIT: {stype} =====")
        runs = []
        for seed in args.seeds:
            if stype == "scaffold":
                tr, va, te = du.scaffold_split(ds["scaffolds"], seed=seed)
            else:
                tr, va, te = du.random_split(n, seed=seed)
            te_frac = ds["labels"][te].mean()
            print(f"[seed {seed}] train={len(tr)} val={len(va)} test={len(te)} "
                  f"test_active_frac={te_frac:.3f}")

            gnn = train_gnn(ds["graphs"], tr, va, te, seed)
            xgbr = train_xgb(ds["fps"], ds["labels"], tr, va, te, seed)
            rf = train_rf(ds["fps"], ds["labels"], tr, va, te, seed)
            print(f"   GNN   AUROC={gnn['auroc']:.4f} AUPRC={gnn['auprc']:.4f}")
            print(f"   XGB   AUROC={xgbr['auroc']:.4f} AUPRC={xgbr['auprc']:.4f}")
            print(f"   RF    AUROC={rf['auroc']:.4f} AUPRC={rf['auprc']:.4f}")
            runs.append({"GNN": gnn, "XGBoost": xgbr, "RandomForest": rf})

        summ = summarize(runs)
        results["splits"][stype] = {"per_seed": runs, "summary": summ}
        print(f"--- {stype} summary ---")
        for m in ("GNN", "XGBoost", "RandomForest"):
            print(f"  {m}: AUROC {summ[f'{m}_auroc_mean']:.4f}+/-{summ[f'{m}_auroc_std']:.4f}  "
                  f"AUPRC {summ[f'{m}_auprc_mean']:.4f}+/-{summ[f'{m}_auprc_std']:.4f}")

    results["runtime_sec"] = time.time() - t0
    out_path = "/home/elrarun/code/research/molgnn/results/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path} ({results['runtime_sec']:.0f}s total)")


if __name__ == "__main__":
    main()
