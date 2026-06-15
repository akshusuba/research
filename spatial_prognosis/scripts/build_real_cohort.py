"""Assemble a tidy per-cell table for the Jackson-Fischer 2020 Basel IMC cohort.

Merges three Zenodo sources (record 4607374 + metadata from 3518284/4607374):
  * Basel_SC_locations.csv        -> per-cell X/Y + core (sample id)
  * Basel_metaclusters.csv        -> per-cell metacluster (marker-derived cell type)
  * Metacluster_annotations.csv   -> metacluster -> named cell type
  * Basel_PatientMetadata.csv     -> per-core grade + overall survival

Writes:
  data/imc/basel_cells.csv.gz     (one row per cell: sample_id, x, y, cell_type)
  data/imc/basel_labels.csv       (one row per core: grade, OSmonth, event)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IMC = os.path.join(HERE, "..", "data", "imc")


def main():
    loc = pd.read_csv(os.path.join(IMC, "extracted", "Basel_SC_locations.csv"))
    mc = pd.read_csv(os.path.join(IMC, "extracted", "Cluster_labels", "Basel_metaclusters.csv"))
    ann = pd.read_csv(os.path.join(IMC, "extracted", "Cluster_labels",
                                   "Metacluster_annotations.csv"), sep=";")
    ann.columns = [c.strip() for c in ann.columns]
    ann["Metacluster"] = ann["Metacluster"].astype(int)
    ct_map = dict(zip(ann["Metacluster"], ann["Cell type"].str.strip()))

    meta = pd.read_csv(os.path.join(IMC, "meta", "Data_publication",
                                    "BaselTMA", "Basel_PatientMetadata.csv"))

    cells = loc.merge(mc, on="id", how="inner")
    cells["cell_type"] = cells["cluster"].map(ct_map)
    cells = cells.rename(columns={"core": "sample_id",
                                  "Location_Center_X": "x",
                                  "Location_Center_Y": "y"})
    cells = cells[["sample_id", "x", "y", "cell_type", "cluster"]].dropna()
    cells["cell_type"] = cells["cell_type"].astype(str)

    # labels per core
    died = meta["Patientstatus"].isin(["death by primary disease", "death"]).astype(int)
    lab = pd.DataFrame({
        "sample_id": meta["core"].astype(str),
        "PID": meta["PID"],
        "grade": meta["grade"],
        "OSmonth": meta["OSmonth"],
        "event": died,            # 1 = died, 0 = alive/censored
    })

    os.makedirs(IMC, exist_ok=True)
    cells.to_csv(os.path.join(IMC, "basel_cells.csv.gz"), index=False, compression="gzip")
    lab.to_csv(os.path.join(IMC, "basel_labels.csv"), index=False)

    print("cells:", cells.shape, "| cores in cells:", cells["sample_id"].nunique())
    print("labels:", lab.shape, "| grade dist:", lab["grade"].value_counts().to_dict())
    print("cells per core: median",
          int(cells.groupby("sample_id").size().median()),
          "max", int(cells.groupby("sample_id").size().max()))
    print("n cell types:", cells["cell_type"].nunique())


if __name__ == "__main__":
    main()
