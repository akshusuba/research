"""Real spatial-omics loader (AnnData -> spatial graph for node classification).

This reads any spatially-resolved dataset in AnnData ``.h5ad`` form that carries
(a) 2D coordinates in ``adata.obsm[spatial_key]`` and (b) a ground-truth domain
label column in ``adata.obs[label_key]``. It standard-preprocesses expression
(normalize -> log1p -> HVG -> PCA) into node features, builds a per-section
spatial kNN graph, and returns the same ``SpatialDataset`` container used by the
synthetic benchmark, so the identical models/splits/trainer apply.

Recommended benchmark: the LIBD **DLPFC** Visium dataset (12 sections, manual
cortical-layer annotations) -- the standard ground truth for spatial-domain
methods. It is distributed via R/Bioconductor, so export it to ``.h5ad`` once:

    # in R
    library(spatialLIBD); library(zellkonverter)
    spe <- fetch_data("spe")               # 12 DLPFC sections, layer labels
    writeH5AD(spe, "dlpfc.h5ad")           # obs: 'layer_guess_reordered', 'sample_id'

then:

    load_h5ad_spatial("dlpfc.h5ad", label_key="layer_guess_reordered",
                      sample_key="sample_id")

Any 10x Visium / MERFISH / Xenium AnnData with a label column works the same way.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from .synthetic import SpatialDataset


def _knn_edges(pos, k):
    n = pos.shape[0]
    k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(pos)
    _, idx = nn.kneighbors(pos)
    src = np.repeat(np.arange(n), k - 1)
    dst = idx[:, 1:].reshape(-1)
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    return np.stack([s, d])


def load_h5ad_spatial(path: str, label_key: str, spatial_key: str = "spatial",
                      sample_key: str | None = None, n_top_genes: int = 2000,
                      n_pcs: int = 50, k_neighbors: int = 6) -> SpatialDataset:
    import scanpy as sc

    adata = sc.read_h5ad(path)

    # Drop cells without a domain label (e.g., unannotated spots).
    labeled = adata.obs[label_key].notna().values
    adata = adata[labeled].copy()

    # Standard expression preprocessing -> PCA node features.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars))
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1))
    feats = np.asarray(adata.obsm["X_pca"], dtype=np.float32)

    # Labels -> integer codes.
    y_cat = adata.obs[label_key].astype("category")
    y = y_cat.cat.codes.values.astype(np.int64)

    # Spatial coords + per-section graph (disjoint union across sections).
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[:, :2]
    if sample_key is not None and sample_key in adata.obs:
        samples = adata.obs[sample_key].astype(str).values
    else:
        samples = np.zeros(adata.n_obs, dtype=str)

    uniq = {s: i for i, s in enumerate(sorted(set(samples)))}
    slice_id = np.array([uniq[s] for s in samples], dtype=np.int64)

    edge_src, edge_dst = [], []
    for s in sorted(set(samples)):
        idx = np.where(samples == s)[0]
        e = _knn_edges(coords[idx], k_neighbors)
        edge_src.append(idx[e[0]])
        edge_dst.append(idx[e[1]])
    edge_index = torch.from_numpy(
        np.stack([np.concatenate(edge_src), np.concatenate(edge_dst)])).long()

    data = Data(x=torch.from_numpy(feats),
                edge_index=edge_index,
                y=torch.from_numpy(y),
                pos=torch.from_numpy(coords),
                num_nodes=feats.shape[0])
    data.slice_id = torch.from_numpy(slice_id)
    n_domains = int(y.max() + 1)
    return SpatialDataset(data=data, n_domains=n_domains)
