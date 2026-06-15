"""Unified training/eval for the spatial GNN and the composition-only baselines.

The GNN trains on the cell graphs (sees arrangement); the baselines train on
per-patient 'bag-of-cells' composition features (blind to arrangement). Same
splits and metrics, so any gap is the value of spatial structure.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .config import ModelConfig, TrainConfig
from .data.synthetic import composition_features
from .metrics import classification_metrics
from .models import GNNGraphClassifier
from .splits import CohortSplit


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_any(name, ds, split: CohortSplit, model_cfg, train_cfg, seed=0):
    name = name.lower()
    if name in ("gnn", "sage", "gin", "gcn"):
        return _train_gnn(name, ds, split, model_cfg, train_cfg, seed)
    if name in ("xgboost", "mlp", "logreg"):
        return _train_composition(name, ds, split, seed)
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Composition-only baselines (blind to arrangement)
# ---------------------------------------------------------------------------
def _train_composition(name, ds, split, seed):
    X = composition_features(ds.graphs)
    y = np.array([int(g.y) for g in ds.graphs])
    Xtr, ytr = X[split.train_idx], y[split.train_idx]
    Xte, yte = X[split.test_idx], y[split.test_idx]
    Xva, yva = X[split.val_idx], y[split.val_idx]

    if name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", n_jobs=0, random_state=seed)
        clf.fit(Xtr, ytr)
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500,
                            random_state=seed)
        clf.fit(Xtr, ytr)
    else:  # logreg
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(Xtr, ytr)

    def ev(Xs, ys):
        score = clf.predict_proba(Xs)[:, 1]
        pred = (score >= 0.5).astype(int)
        return classification_metrics(ys, pred, score)

    return {"model": name, "val": ev(Xva, yva), "test": ev(Xte, yte)}


# ---------------------------------------------------------------------------
# Spatial GNN (reads arrangement)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _eval_gnn(model, loader, device):
    model.eval()
    ys, preds, scores = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        prob = logits.softmax(dim=-1)[:, 1]
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        scores.append(prob.cpu().numpy())
        ys.append(batch.y.cpu().numpy())
    return classification_metrics(np.concatenate(ys), np.concatenate(preds),
                                  np.concatenate(scores))


def _train_gnn(name, ds, split, model_cfg: ModelConfig, train_cfg: TrainConfig, seed):
    set_seed(seed)
    device = torch.device(train_cfg.device)
    cfg = copy.deepcopy(model_cfg)
    if name in ("sage", "gin", "gcn"):
        cfg.encoder = name

    graphs = ds.graphs
    tr = [graphs[i] for i in split.train_idx]
    va = [graphs[i] for i in split.val_idx]
    te = [graphs[i] for i in split.test_idx]
    tr_loader = DataLoader(tr, batch_size=train_cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=train_cfg.batch_size)
    te_loader = DataLoader(te, batch_size=train_cfg.batch_size)

    model = GNNGraphClassifier(split.num_features, split.num_classes, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)

    best_val, best_state, patience = -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(train_cfg.epochs):
        model.train()
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            opt.step()
        val_m = _eval_gnn(model, va_loader, device)
        if val_m["macro_f1"] > best_val:
            best_val = val_m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.patience:
                break

    model.load_state_dict(best_state)
    return {"model": name, "val": _eval_gnn(model, va_loader, device),
            "test": _eval_gnn(model, te_loader, device)}
