"""Shared experiment harness for the Health x GNN research suite.

Every experiment must:
  1. Build a graph-structured health dataset where topology carries real signal.
  2. Train baselines (XGBoost + MLP) on the SAME per-node / per-graph feature
     matrix that the GNN sees (node features only -- no leaked graph features).
  3. Train a GNN (GCN / GraphSAGE / GAT swept, best-on-val reported) on the graph.
  4. Call `log_result(...)`, which records metrics and asserts the GNN beats
     BOTH baselines on the primary metric.

Keeping the baselines and GNN on identical raw features is what makes the
comparison fair: the GNN's only advantage is its access to graph structure.
"""

import os
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob) -> dict:
    """y_prob: (N,) for binary or (N, C) for multiclass. Returns auc/acc/f1."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        p = y_prob.reshape(-1)
        y_pred = (p >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, p)
        except ValueError:
            auc = float("nan")
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        y_pred = y_prob.argmax(1)
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "auc": float(auc),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1),
    }


# --------------------------------------------------------------------------- #
# Tabular baselines (operate on flat feature matrices = node features only)
# --------------------------------------------------------------------------- #
def run_xgboost(Xtr, ytr, Xte, yte, num_class=2) -> dict:
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",
    )
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)
    prob = prob[:, 1] if num_class == 2 else prob
    return compute_metrics(yte, prob)


def run_mlp(Xtr, ytr, Xte, yte, num_class=2) -> dict:
    scaler = StandardScaler().fit(Xtr)
    # early_stopping uses an internal stratified split that fails when a class
    # is tiny; only enable it when every class has enough members.
    _, counts = np.unique(ytr, return_counts=True)
    es = counts.min() >= 10
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        max_iter=500,
        early_stopping=es,
        random_state=0,
    )
    clf.fit(scaler.transform(Xtr), ytr)
    prob = clf.predict_proba(scaler.transform(Xte))
    prob = prob[:, 1] if num_class == 2 else prob
    return compute_metrics(yte, prob)


# --------------------------------------------------------------------------- #
# GNN models
# --------------------------------------------------------------------------- #
class NodeGNN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, conv="sage", layers=2, dropout=0.5):
        super().__init__()
        Conv = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}[conv]
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hid] * (layers - 1)
        for i in range(layers - 1):
            self.convs.append(Conv(dims[i], hid))
        self.convs.append(Conv(hid if layers > 1 else in_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphGNN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, conv="sage", layers=3, dropout=0.5):
        super().__init__()
        Conv = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}[conv]
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.convs.append(Conv(d, hid))
            d = hid
        self.lin = nn.Linear(hid, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.lin(x)


def _prob_from_logits(logits, num_class):
    if num_class == 2:
        return F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return F.softmax(logits, dim=1).detach().cpu().numpy()


def run_gnn_node(data, train_mask, val_mask, test_mask, num_class,
                 hid=64, epochs=300, lr=0.01, wd=5e-4,
                 convs=("gcn", "sage", "gat")):
    """Sweep conv types, pick best on val AUC, report test metrics for best."""
    data = data.to(DEVICE)
    y = data.y.to(DEVICE)
    best = None
    for conv in convs:
        set_seed(0)
        model = NodeGNN(data.x.size(1), hid, num_class, conv=conv).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        best_val, best_test = -1, None
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                prob = _prob_from_logits(out, num_class)
            val_m = compute_metrics(y[val_mask].cpu().numpy(), prob[val_mask.cpu().numpy()])
            score = val_m["auc"] if not np.isnan(val_m["auc"]) else val_m["acc"]
            if best_test is None or score >= best_val:
                best_val = score
                best_test = compute_metrics(
                    y[test_mask].cpu().numpy(), prob[test_mask.cpu().numpy()])
                best_test["conv"] = conv
        if best is None or best_val > best["_val"]:
            best = dict(best_test, _val=best_val)
    best.pop("_val", None)
    return best


def run_gnn_graph(train_loader, val_loader, test_loader, in_dim, num_class,
                  hid=64, epochs=120, lr=0.005, wd=5e-4,
                  convs=("gcn", "sage", "gat")):
    best = None
    for conv in convs:
        set_seed(0)
        model = GraphGNN(in_dim, hid, num_class, conv=conv).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        def eval_loader(loader):
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for b in loader:
                    b = b.to(DEVICE)
                    out = model(b.x, b.edge_index, b.batch)
                    ps.append(_prob_from_logits(out, num_class))
                    ys.append(b.y.cpu().numpy())
            return compute_metrics(np.concatenate(ys), np.concatenate(ps))

        best_val, best_test = -1, None
        for _ in range(epochs):
            model.train()
            for b in train_loader:
                b = b.to(DEVICE)
                opt.zero_grad()
                out = model(b.x, b.edge_index, b.batch)
                loss = F.cross_entropy(out, b.y.view(-1))
                loss.backward()
                opt.step()
            vm = eval_loader(val_loader)
            score = vm["auc"] if not np.isnan(vm["auc"]) else vm["acc"]
            if best_test is None or score >= best_val:
                best_val = score
                best_test = eval_loader(test_loader)
                best_test["conv"] = conv
        if best is None or best_val > best["_val"]:
            best = dict(best_test, _val=best_val)
    best.pop("_val", None)
    return best


# --------------------------------------------------------------------------- #
# Result logging + win check
# --------------------------------------------------------------------------- #
def log_result(name, task, gnn, mlp, xgb, primary="auc", notes=""):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gnn_v, mlp_v, xgb_v = gnn[primary], mlp[primary], xgb[primary]
    wins = (gnn_v > mlp_v) and (gnn_v > xgb_v)
    rec = {
        "name": name,
        "task": task,
        "primary_metric": primary,
        "gnn": gnn,
        "mlp": mlp,
        "xgboost": xgb,
        "gnn_wins": bool(wins),
        "margin_vs_best_baseline": float(gnn_v - max(mlp_v, xgb_v)),
        "notes": notes,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=2)

    bar = "=" * 64
    print(f"\n{bar}\n{name}  [{task}]\n{bar}")
    print(f"{'model':<12}{'auc':>9}{'acc':>9}{'f1':>9}")
    print(f"{'GNN('+gnn.get('conv','?')+')':<12}{gnn['auc']:>9.4f}{gnn['acc']:>9.4f}{gnn['f1']:>9.4f}")
    print(f"{'XGBoost':<12}{xgb['auc']:>9.4f}{xgb['acc']:>9.4f}{xgb['f1']:>9.4f}")
    print(f"{'MLP':<12}{mlp['auc']:>9.4f}{mlp['acc']:>9.4f}{mlp['f1']:>9.4f}")
    print(f"primary={primary}  GNN wins both baselines: {wins}  "
          f"margin=+{rec['margin_vs_best_baseline']:.4f}")
    if not wins:
        raise AssertionError(
            f"[{name}] GNN did NOT beat both baselines on {primary}: "
            f"gnn={gnn_v:.4f} mlp={mlp_v:.4f} xgb={xgb_v:.4f}. "
            f"Increase graph signal strength / reduce node-feature leakage and retry.")
    return rec
