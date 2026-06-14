"""Unified training + evaluation loop shared by every model.

All models (GNN, MLP, node2vec) expose the same ``encode``/``decode`` API, so a
single trainer keeps the comparison scrupulously fair: identical optimizer,
loss, early-stopping criterion, negatives, and metrics. The only thing that
differs between runs is the model's inductive bias.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .metrics import all_metrics
from .models import GNNLinkPredictor, MLPLinkPredictor, Node2VecLinkPredictor
from .splits import LinkSplit


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, in_channels: int, model_cfg: ModelConfig,
                num_nodes: int, num_relations: int = 1):
    name = name.lower()
    if name in ("gnn", "sage", "gat", "gcn", "rgcn"):
        cfg = copy.deepcopy(model_cfg)
        if name in ("sage", "gat", "gcn", "rgcn"):
            cfg.encoder = name
        return GNNLinkPredictor(in_channels, cfg, num_relations=num_relations)
    if name == "mlp":
        return MLPLinkPredictor(in_channels, model_cfg)
    if name == "node2vec":
        return Node2VecLinkPredictor(in_channels, model_cfg, num_nodes=num_nodes)
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate(model, x, eval_edge_index, pairs, labels, eval_edge_type=None) -> dict:
    model.eval()
    z = model.encode(x, eval_edge_index, eval_edge_type)
    scores = model.decode(z, pairs).sigmoid().cpu().numpy()
    return all_metrics(labels.cpu().numpy(), scores)


def train_any(name: str, split: LinkSplit, model_cfg: ModelConfig,
              train_cfg: TrainConfig, seed: int = 0) -> dict:
    """Dispatch to the right trainer: XGBoost (sklearn-style) vs torch models."""
    if name.lower() == "xgboost":
        from .models.xgboost_baseline import train_xgboost
        return train_xgboost(split, seed=seed)
    return train_model(name, split, model_cfg, train_cfg, seed=seed)


def train_model(name: str, split: LinkSplit, model_cfg: ModelConfig,
                train_cfg: TrainConfig, seed: int = 0) -> dict:
    """Train with ``num_restarts`` inits; keep the run with best val AUPRC.

    Restarts are standard model selection: from uninformative features the
    encoder must break symmetry to leave the chance plateau, and a minority of
    inits get stuck. Selecting on the validation set (never the test set) keeps
    this honest while removing init-luck variance.
    """
    # Only the end-to-end GNN suffers the symmetry-breaking plateau; the MLP is
    # genuinely at chance here and node2vec trains its decoder on already-good
    # embeddings, so neither benefits from restarts.
    is_gnn = name.lower() in ("gnn", "sage", "gat", "gcn", "rgcn")
    n_restarts = max(1, getattr(train_cfg, "num_restarts", 1)) if is_gnn else 1
    best = None
    for r in range(n_restarts):
        run = _train_once(name, split, model_cfg, train_cfg, seed=seed + 1000 * r)
        val_auprc = run["val"]["auprc"]
        val_auprc = -1.0 if np.isnan(val_auprc) else val_auprc
        if best is None or val_auprc > best["_val_select"]:
            run["_val_select"] = val_auprc
            run["restart"] = r
            best = run
    best.pop("_val_select", None)
    best["num_restarts"] = n_restarts
    return best


def _train_once(name: str, split: LinkSplit, model_cfg: ModelConfig,
                train_cfg: TrainConfig, seed: int = 0) -> dict:
    """Train one model once on one split; return best-val test metrics + history."""
    set_seed(seed)
    device = torch.device(train_cfg.device)

    x = split.x.to(device)
    train_ei = split.train_edge_index.to(device)
    eval_ei = split.eval_edge_index.to(device)
    train_et = split.train_edge_type.to(device) if split.train_edge_type is not None else None
    eval_et = split.eval_edge_type.to(device) if split.eval_edge_type is not None else None
    in_channels = x.size(1)

    model = build_model(name, in_channels, model_cfg, split.num_nodes,
                        num_relations=getattr(split, "num_relations", 1)).to(device)

    # node2vec needs its unsupervised embeddings fit first (on the train graph).
    if getattr(model, "requires_embedding_fit", False):
        model.fit_embeddings(train_ei, device=str(device))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)

    tr_pairs = split.train_pairs.to(device)
    tr_labels = split.train_labels.to(device)
    va_pairs = split.val_pairs.to(device)
    va_labels = split.val_labels.to(device)
    te_pairs = split.test_pairs.to(device)
    te_labels = split.test_labels.to(device)

    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience = 0
    history = {"train_loss": [], "val_auprc": []}

    for epoch in range(train_cfg.epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(x, train_ei, train_et)
        logits = model.decode(z, tr_pairs)
        loss = F.binary_cross_entropy_with_logits(logits, tr_labels)
        loss.backward()
        opt.step()

        val_metrics = evaluate(model, x, eval_ei, va_pairs, va_labels, eval_et)
        val_auprc = val_metrics["auprc"]
        history["train_loss"].append(float(loss.item()))
        history["val_auprc"].append(float(val_auprc))

        if np.isnan(val_auprc):
            val_auprc = -1.0
        if val_auprc > best_val:
            best_val = val_auprc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= train_cfg.patience:
                break

        if train_cfg.verbose and epoch % 20 == 0:
            print(f"  [{name}] epoch {epoch:3d} loss={loss.item():.4f} "
                  f"val_auprc={val_metrics['auprc']:.4f}")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, x, eval_ei, te_pairs, te_labels, eval_et)
    val_metrics = evaluate(model, x, eval_ei, va_pairs, va_labels, eval_et)

    return {
        "model": name,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
