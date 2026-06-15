"""Tuned XGBoost tabular baseline (the structure-blind control to beat).

Features per (drug, disease) pair = concatenation of the two nodes' shared text
features -- identical inputs to the GNN, so the comparison isolates topology.
XGBoost cannot traverse the KG, so any GNN gain over a well-tuned XGBoost is
attributable to multi-hop graph structure.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from oncorepurpose.evaluation.metrics import compute_all_metrics
from oncorepurpose.evaluation.splits import SplitData


def _pair_features(data: HeteroData, et, eli: torch.Tensor) -> np.ndarray:
    s_t, _, d_t = et
    xs = data[s_t].x[eli[0]].cpu().numpy()
    xd = data[d_t].x[eli[1]].cpu().numpy()
    return np.concatenate([xs, xd], axis=1)


def run_xgboost(
    split: SplitData, data: HeteroData, seed: int = 0,
    n_estimators: int = 400, max_depth: int = 6, lr: float = 0.1,
    tune: bool = False, n_trials: int = 20,
) -> Dict[str, float]:
    import xgboost as xgb

    et = split.target_edge_type
    Xtr = _pair_features(data, et, split.train_label_index)
    ytr = split.train_label.cpu().numpy()
    Xva = _pair_features(data, et, split.val_label_index)
    yva = split.val_label.cpu().numpy()
    Xte = _pair_features(data, et, split.test_label_index)
    yte = split.test_label.cpu().numpy()

    # Early stopping keeps tuning fast: estimators are capped automatically on the
    # validation set rather than always training the full count.
    base = dict(
        n_estimators=600, max_depth=max_depth, learning_rate=lr,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        tree_method="hist", random_state=seed, n_jobs=-1, early_stopping_rounds=30,
    )

    if tune:
        import optuna
        from sklearn.metrics import roc_auc_score

        def objective(trial):
            p = dict(
                n_estimators=600,
                max_depth=trial.suggest_int("max_depth", 3, 9),
                learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
                eval_metric="logloss", tree_method="hist", random_state=seed,
                n_jobs=-1, early_stopping_rounds=30,
            )
            m = xgb.XGBClassifier(**p)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        base.update(study.best_params)

    model = xgb.XGBClassifier(**base)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    scores = model.predict_proba(Xte)[:, 1]
    return compute_all_metrics(yte, scores)
