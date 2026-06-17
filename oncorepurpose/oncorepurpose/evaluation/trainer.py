"""Shared training/eval for GNN (neighbor sampling), FeatureMLP, and DistMult KGE."""

from __future__ import annotations

import copy
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from oncorepurpose.evaluation.metrics import compute_all_metrics
from oncorepurpose.evaluation.splits import SplitData
from oncorepurpose.models import DistMultKGE, FeatureMLP, HeteroGNN


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def _eval_encoder(model, base: HeteroData, et, eli: torch.Tensor, labels: torch.Tensor, device) -> Dict[str, float]:
    model.eval()
    z = model.encode(base)
    scores = torch.sigmoid(model.decode(z, et, eli)).cpu()
    return compute_all_metrics(labels.cpu(), scores)


def train_gnn(
    model: HeteroGNN, split: SplitData, device: torch.device,
    epochs: int = 50, patience: int = 10, lr: float = 5e-3, weight_decay: float = 1e-5,
    verbose: bool = False,
) -> HeteroGNN:
    """Full-batch training: message-pass over the whole graph each step.

    (Neighbor sampling would need pyg-lib/torch-sparse; full-batch is ~3s/step on
    this graph and avoids that dependency.)
    """
    model = model.to(device)
    et = split.target_edge_type
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y = split.train_label.float().to(device)

    best_val, best_state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(split.base)
        pred = model.decode(z, et, split.train_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
        val = _eval_encoder(model, split.base, et, split.val_label_index, split.val_label, device)["auroc"]
        if val > best_val:
            best_val, wait = val, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and (epoch + 1) % 10 == 0:
            print(f"      gnn epoch {epoch+1} val_auroc={val:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_gnn_joint(
    model: HeteroGNN, split: SplitData, device: torch.device,
    mech, decoys, lam: float = 1.0, n_neg: int = 8, mech_batch: int = 256,
    epochs: int = 60, lr: float = 5e-3, weight_decay: float = 1e-5, verbose: bool = False,
) -> HeteroGNN:
    """Joint training: link BCE + lambda * InfoNCE mechanism loss.

    The mechanism term (covered TRAINING pairs only) pushes the true DrugMechDB
    bridge gene above degree-matched decoys, using the SAME encoder embeddings.
    Trained for a fixed number of epochs (no early stopping) so the mechanism head
    co-converges with the link objective rather than being cut off by link val.
    """
    model = model.to(device)
    et = split.target_edge_type
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y = split.train_label.float().to(device)
    md, mc, mg = mech.drug, mech.dis, mech.gene
    M = int(md.numel())

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(split.base)
        link_loss = F.binary_cross_entropy_with_logits(model.decode(z, et, split.train_label_index), y)
        mech_loss = torch.tensor(0.0, device=device)
        if M > 0 and lam > 0:
            b = min(mech_batch, M)
            idx = torch.randint(0, M, (b,))
            bd, bc, bg = md[idx], mc[idx], mg[idx]
            neg = torch.tensor(
                [decoys.sample(int(bg[j]), {int(bg[j])}, n_neg) for j in range(b)],
                dtype=torch.long,
            )                                            # [b, n_neg]
            cand = torch.cat([bg.view(-1, 1), neg], dim=1)  # [b, 1+n_neg], col0 = positive
            K = cand.size(1)
            flat_d = bd.view(-1, 1).expand(-1, K).reshape(-1)
            flat_c = bc.view(-1, 1).expand(-1, K).reshape(-1)
            flat_g = cand.reshape(-1)
            logits = model.score_mechanism(z, flat_d, flat_g, flat_c).view(b, K)
            mech_loss = F.cross_entropy(logits, torch.zeros(b, dtype=torch.long, device=logits.device))
        loss = link_loss + lam * mech_loss
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"      joint epoch {epoch+1} link={link_loss.item():.3f} mech={float(mech_loss):.3f}")
    return model


def train_mlp(
    model: FeatureMLP, split: SplitData, device: torch.device,
    epochs: int = 200, patience: int = 20, lr: float = 5e-3, weight_decay: float = 1e-5, verbose: bool = False,
) -> FeatureMLP:
    model = model.to(device)
    et = split.target_edge_type
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y = split.train_label.float().to(device)
    best_val, best_state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(split.base)
        pred = model.decode(z, et, split.train_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
        val = _eval_encoder(model, split.base, et, split.val_label_index, split.val_label, device)["auroc"]
        if val > best_val:
            best_val, wait = val, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _eval_kge(model: DistMultKGE, eli: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    model.eval()
    scores = torch.sigmoid(model.score(eli)).cpu()
    return compute_all_metrics(labels.cpu(), scores)


def train_kge(
    model: DistMultKGE, split: SplitData, device: torch.device,
    epochs: int = 300, patience: int = 30, lr: float = 1e-2, weight_decay: float = 1e-6, verbose: bool = False,
) -> DistMultKGE:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y = split.train_label.float().to(device)
    best_val, best_state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model.score(split.train_label_index)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
        val = _eval_kge(model, split.val_label_index, split.val_label)["auroc"]
        if val > best_val:
            best_val, wait = val, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, split: SplitData, device: torch.device, which: str = "test") -> Dict[str, float]:
    eli = getattr(split, f"{which}_label_index")
    lab = getattr(split, f"{which}_label")
    if isinstance(model, DistMultKGE):
        return _eval_kge(model, eli, lab)
    return _eval_encoder(model, split.base, split.target_edge_type, eli, lab, device)
