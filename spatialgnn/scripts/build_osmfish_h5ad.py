#!/usr/bin/env python
"""Build the osmFISH mouse somatosensory cortex benchmark into a single .h5ad.

Source: Codeluppi et al. 2018 (Linnarsson Lab), osmFISH -- a single-molecule FISH
spatial transcriptomics assay at single-cell resolution. Unlike Visium (spots),
each node here is a single cell with only 33 measured genes, so per-cell features
are deliberately weak -- a strong test of whether neighbourhood aggregation
recovers the spatial DOMAIN.

Ground-truth labels: ``Region`` (cortical layers L1/Pia, L2-3, L4, L5, L6, plus
White matter, Ventricle, Hippocampus, Internal Capsule Caudoputamen). 'Excluded'
cells are dropped. Single tissue section -> use within-section spatial splits.

Output: data/osmfish.h5ad with
  obs['Region']     -- region/layer ground-truth domain labels
  obs['sample_id']  -- single section id
  obsm['spatial']   -- 2D coordinates [X, Y]
"""
import os

import anndata as ad
import loompy
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOM = os.path.join(HERE, "data", "osmfish.loom")
OUT = os.path.join(HERE, "data", "osmfish.h5ad")


def main():
    ds = loompy.connect(LOOM)
    X = ds[:, :].T.astype(np.float32)  # loom is genes x cells -> cells x genes
    gene_attr = "Gene" if "Gene" in ds.ra.keys() else list(ds.ra.keys())[0] if ds.ra.keys() else None
    genes = (np.array(ds.ra[gene_attr]).astype(str) if gene_attr
             else np.array([f"gene_{i}" for i in range(ds.shape[0])]))
    obs = pd.DataFrame({k: np.array(ds.ca[k]) for k in ds.ca.keys()})
    ds.close()

    obs["Region"] = obs["Region"].astype(str)
    adata = ad.AnnData(X=X, obs=obs.reset_index(drop=True),
                       var=pd.DataFrame(index=[str(g) for g in genes]))
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.obsm["spatial"] = adata.obs[["X", "Y"]].to_numpy(dtype=np.float32)
    adata.obs["sample_id"] = "osmfish_cortex"

    keep = ~adata.obs["Region"].isin(["Excluded", "", "nan", "None"])
    adata = adata[keep].copy()

    print(f"cells: {adata.n_obs}, genes: {adata.n_vars}")
    print("regions:\n", adata.obs["Region"].value_counts())
    adata.write_h5ad(OUT)
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
