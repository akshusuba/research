"""Real imaging-mass-cytometry (IMC) cohort loader for spatial prognosis.

Builds per-patient cell graphs from a long-form single-cell table plus a
per-sample clinical table, so the same models/splits/trainer used on the
synthetic cohort apply unchanged.

Recommended cohort: the **Jackson-Fischer 2020 breast cancer IMC** dataset
(Basel + Zurich, ~700 patients; 37 protein markers; tumor grade + survival).
Raw data: Zenodo ``10.5281/zenodo.3518284`` (single-cell CSVs + metadata). It is
also wrapped by the R/Bioconductor ``imcdatasets`` package, but the Zenodo CSVs
are directly usable from Python.

Expected inputs (column names are configurable):
  * cell table  : one row per cell with [sample_id, x, y, cell_type, marker_1...]
  * clinical map: {sample_id -> label}, e.g. tumor grade (1/2/3) or
                  binarized survival (short/long).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from .synthetic import CohortDataset


def _knn_edges(pos, k):
    n = pos.shape[0]
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(pos)
    _, idx = nn.kneighbors(pos)
    src = np.repeat(np.arange(n), k - 1)
    dst = idx[:, 1:].reshape(-1)
    s = np.concatenate([src, dst]); d = np.concatenate([dst, src])
    return torch.from_numpy(np.stack([s, d])).long()


def build_cohort_from_tables(
    cells: pd.DataFrame, labels: Dict[str, int], *,
    sample_col: str = "sample_id", x_col: str = "x", y_col: str = "y",
    celltype_col: Optional[str] = "cell_type",
    marker_cols: Optional[List[str]] = None, k_neighbors: int = 6,
    min_cells: int = 50, standardize: bool = True,
) -> CohortDataset:
    """Assemble a `CohortDataset` of per-patient spatial cell graphs.

    Node features = one-hot(cell_type) (if given) concatenated with marker
    columns. Edges = spatial kNN within each sample. Graph label = labels[sample].
    """
    if marker_cols is None:
        reserved = {sample_col, x_col, y_col, celltype_col}
        marker_cols = [c for c in cells.columns
                       if c not in reserved and pd.api.types.is_numeric_dtype(cells[c])]

    # standardize markers globally
    M = cells[marker_cols].to_numpy(dtype=np.float32)
    if standardize and M.shape[1] > 0:
        M = (M - M.mean(0)) / (M.std(0) + 1e-8)
    cells = cells.copy()
    cells[marker_cols] = M

    # cell-type vocabulary
    ct_vocab = None
    if celltype_col is not None and celltype_col in cells.columns:
        ct_vocab = {c: i for i, c in enumerate(sorted(cells[celltype_col].astype(str).unique()))}

    graphs: List[Data] = []
    for sample, grp in cells.groupby(sample_col):
        sample = str(sample)
        if sample not in labels:
            continue
        if len(grp) < min_cells:
            continue
        pos = grp[[x_col, y_col]].to_numpy(dtype=np.float32)
        feats = [grp[marker_cols].to_numpy(dtype=np.float32)] if marker_cols else []
        if ct_vocab is not None:
            codes = grp[celltype_col].astype(str).map(ct_vocab).to_numpy()
            onehot = np.zeros((len(grp), len(ct_vocab)), dtype=np.float32)
            onehot[np.arange(len(grp)), codes] = 1.0
            feats = [onehot] + feats
        x = np.concatenate(feats, axis=1).astype(np.float32)
        g = Data(x=torch.from_numpy(x), edge_index=_knn_edges(pos, k_neighbors),
                 y=torch.tensor([int(labels[sample])], dtype=torch.long),
                 pos=torch.from_numpy(pos))
        graphs.append(g)

    if not graphs:
        raise ValueError("No patient graphs built -- check sample_col / labels keys.")
    num_classes = int(max(int(g.y) for g in graphs) + 1)
    return CohortDataset(graphs=graphs, num_classes=num_classes,
                         num_features=graphs[0].x.size(1))


def binarize_survival(times, events, cutoff_months: float = 60.0):
    """Binary outcome: 1 = survived past cutoff, 0 = died before cutoff.

    Patients censored before the cutoff (event=0, time<cutoff) are dropped as
    their outcome is unknown.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    label = np.full(len(times), -1, dtype=int)
    label[times >= cutoff_months] = 1
    label[(times < cutoff_months) & (events == 1)] = 0
    return label  # -1 entries should be filtered out by the caller
