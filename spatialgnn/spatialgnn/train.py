"""Unified node-classification trainer shared by GNN and MLP.

Identical optimizer, loss, early-stopping criterion, and metrics across models,
so the only thing that differs is whether the model can see the spatial graph.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .metrics import classification_metrics
from .models import GNNNodeClassifier, MLPNodeClassifier
from .splits import SpatialSplit


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, in_channels: int, num_classes: int,
                model_cfg: ModelConfig):
    name = name.lower()
    if name in ("gnn", "sage", "gat", "gcn"):
        cfg = copy.deepcopy(model_cfg)
        if name in ("sage", "gat", "gcn"):
            cfg.encoder = name
        return GNNNodeClassifier(in_channels, num_classes, cfg)
    if name == "mlp":
        return MLPNodeClassifier(in_channels, num_classes, model_cfg)
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def _eval(model, x, edge_index, y, mask) -> dict:
    model.eval()
    logits = model(x, edge_index)
    pred = logits[mask].argmax(dim=-1).cpu().numpy()
    return classification_metrics(y[mask].cpu().numpy(), pred)


def train_any(name, split, model_cfg, train_cfg, seed=0):
    """Dispatch to XGBoost (sklearn-style) or the torch trainer."""
    if name.lower() == "xgboost":
        from .models.xgboost_baseline import train_xgboost
        return train_xgboost(split, seed=seed)
    return train_model(name, split, model_cfg, train_cfg, seed=seed)


def train_model(name: str, split: SpatialSplit, model_cfg: ModelConfig,
                train_cfg: TrainConfig, seed: int = 0) -> dict:
    set_seed(seed)
    device = torch.device(train_cfg.device)
    x = split.x.to(device)
    edge_index = split.edge_index.to(device)
    y = split.y.to(device)
    tr, va, te = (split.train_mask.to(device), split.val_mask.to(device),
                  split.test_mask.to(device))

    model = build_model(name, x.size(1), split.num_classes, model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)

    best_val, best_state, best_epoch, patience = -1.0, copy.deepcopy(model.state_dict()), 0, 0
    history = {"train_loss": [], "val_f1": []}

    for epoch in range(train_cfg.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[tr], y[tr])
        loss.backward()
        opt.step()

        val_m = _eval(model, x, edge_index, y, va)
        history["train_loss"].append(float(loss.item()))
        history["val_f1"].append(val_m["macro_f1"])
        if val_m["macro_f1"] > best_val:
            best_val = val_m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.patience:
                break
        if train_cfg.verbose and epoch % 25 == 0:
            print(f"  [{name}] ep{epoch:3d} loss={loss.item():.3f} val_f1={val_m['macro_f1']:.3f}")

    model.load_state_dict(best_state)
    return {
        "model": name, "best_epoch": best_epoch,
        "val": _eval(model, x, edge_index, y, va),
        "test": _eval(model, x, edge_index, y, te),
        "history": history,
    }
