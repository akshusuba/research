#!/usr/bin/env python
"""Assemble the 12-section LIBD DLPFC Visium benchmark into a single .h5ad.

Source: HuggingFace dataset ``ylu99/Gres`` (DLPFC/<section>/), which mirrors the
spatialLIBD release. For each of the 12 sections we read the 10x count matrix
(``filtered_feature_bc_matrix.h5``) and join the per-spot ``metadata.tsv`` on
barcode to attach the manual cortical-layer annotation
(``layer_guess_reordered``) and pixel coordinates (imagerow/imagecol).

Output: data/dlpfc.h5ad with
  obs['layer_guess_reordered']  -- L1..L6 / WM ground-truth domain labels
  obs['sample_id']              -- section id (for cross-section splits)
  obsm['spatial']               -- 2D coordinates [imagerow, imagecol]
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "dlpfc_raw")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "dlpfc.h5ad")

SECTIONS = ["151507", "151508", "151509", "151510",
            "151669", "151670", "151671", "151672",
            "151673", "151674", "151675", "151676"]


def load_section(s):
    h5 = os.path.join(RAW, s, "filtered_feature_bc_matrix.h5")
    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()

    meta = pd.read_csv(os.path.join(RAW, s, "metadata.tsv"), sep="\t")
    meta = meta.set_index("barcode")

    # Align metadata to the count-matrix barcodes (obs index).
    common = adata.obs_names.intersection(meta.index)
    adata = adata[common].copy()
    meta = meta.loc[adata.obs_names]

    adata.obs["layer_guess_reordered"] = meta["layer_guess_reordered"].values
    adata.obs["sample_id"] = s
    adata.obsm["spatial"] = meta[["imagerow", "imagecol"]].to_numpy(dtype=np.float32)
    # Make barcodes globally unique across sections before concatenation.
    adata.obs_names = [f"{s}_{b}" for b in adata.obs_names]
    return adata


def main():
    parts = []
    for s in SECTIONS:
        a = load_section(s)
        n_lab = a.obs["layer_guess_reordered"].notna().sum()
        print(f"{s}: {a.n_obs} spots, {a.n_vars} genes, {n_lab} labeled, "
              f"layers={sorted(a.obs['layer_guess_reordered'].dropna().unique())}")
        parts.append(a)

    combined = ad.concat(parts, join="inner", merge="same")
    combined.obs_names_make_unique()
    print("\nCombined:", combined.n_obs, "spots,", combined.n_vars, "genes")
    print("Sections:", combined.obs["sample_id"].value_counts().to_dict())
    lab = combined.obs["layer_guess_reordered"]
    print("Labeled spots:", int(lab.notna().sum()), "/", combined.n_obs)
    print("Label distribution:\n", lab.value_counts(dropna=False))

    combined.write_h5ad(OUT)
    print("\nSaved ->", OUT)


if __name__ == "__main__":
    main()
