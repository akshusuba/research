"""Build a PyG HeteroData object from the PrimeKG kg.csv edge list.

PrimeKG `kg.csv` columns:
    relation, display_relation, x_index, x_id, x_type, x_name, x_source,
    y_index, y_id, y_type, y_name, y_source

We construct a heterogeneous graph with per-type contiguous node indices,
preserve node ids/names (for features + interpretation), normalize type and
relation names for PyG, and flag oncology (neoplasm) diseases.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import (
    DISEASE_TYPE,
    HETERODATA_PT,
    ONCOLOGY_KEYWORDS,
    PRIMEKG_KG_CSV,
)


def _norm_type(t: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(t).strip().lower()).strip("_")


def _norm_rel(r: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(r).strip().lower()).strip("_")


def _is_oncology(name: str) -> bool:
    n = str(name).lower()
    return any(k in n for k in ONCOLOGY_KEYWORDS)


def build_hetero_from_primekg(
    kg_csv: Path = PRIMEKG_KG_CSV,
    output_path: Path = HETERODATA_PT,
    save: bool = True,
) -> HeteroData:
    """Parse PrimeKG into a PyG HeteroData and optionally cache it."""
    kg_csv = Path(kg_csv)
    if not kg_csv.exists():
        raise FileNotFoundError(
            f"PrimeKG not found at {kg_csv}. Run oncorepurpose.data.download first."
        )

    print(f"Loading {kg_csv} ...")
    df = pd.read_csv(kg_csv, dtype=str, low_memory=False)
    print(f"  {len(df):,} raw edges")

    df["x_type_n"] = df["x_type"].map(_norm_type)
    df["y_type_n"] = df["y_type"].map(_norm_type)
    df["rel_n"] = df["relation"].map(_norm_rel)

    # Assign a stable per-type local index to every unique node id.
    node_id_to_idx: Dict[str, Dict[str, int]] = defaultdict(dict)
    node_names: Dict[str, List[str]] = defaultdict(list)
    node_ids: Dict[str, List[str]] = defaultdict(list)

    def _register(node_type: str, node_id: str, node_name: str) -> int:
        idx_map = node_id_to_idx[node_type]
        if node_id not in idx_map:
            idx_map[node_id] = len(idx_map)
            node_names[node_type].append("" if node_name is None else str(node_name))
            node_ids[node_type].append(str(node_id))
        return idx_map[node_id]

    # First pass: register all nodes (vectorized per endpoint).
    for side in ("x", "y"):
        sub = df[[f"{side}_type_n", f"{side}_id", f"{side}_name"]].drop_duplicates()
        for t, i, nm in sub.itertuples(index=False):
            _register(t, i, nm)

    data = HeteroData()
    for ntype, idx_map in node_id_to_idx.items():
        data[ntype].num_nodes = len(idx_map)
        data[ntype].node_ids = node_ids[ntype]
        data[ntype].node_names = node_names[ntype]

    # Second pass: build edge_index per (src_type, relation, dst_type).
    edge_buckets: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    for x_t, x_id, y_t, y_id, rel in df[
        ["x_type_n", "x_id", "y_type_n", "y_id", "rel_n"]
    ].itertuples(index=False):
        s = node_id_to_idx[x_t][x_id]
        d = node_id_to_idx[y_t][y_id]
        edge_buckets[(x_t, rel, y_t)].append((s, d))

    for (s_t, rel, d_t), pairs in edge_buckets.items():
        ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[(s_t, rel, d_t)].edge_index = ei

    # Flag oncology diseases for evaluation focus.
    if DISEASE_TYPE in data.node_types:
        names = data[DISEASE_TYPE].node_names
        mask = torch.tensor([_is_oncology(n) for n in names], dtype=torch.bool)
        data[DISEASE_TYPE].is_oncology = mask
        print(f"  oncology diseases: {int(mask.sum())} / {len(names)}")

    print("HeteroData summary:")
    total_nodes = sum(int(data[t].num_nodes) for t in data.node_types)
    total_edges = sum(int(data[et].edge_index.size(1)) for et in data.edge_types)
    print(f"  node types: {len(data.node_types)} ({total_nodes:,} nodes)")
    print(f"  edge types: {len(data.edge_types)} ({total_edges:,} edges)")

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        print(f"Saved HeteroData to {output_path}")

    return data


if __name__ == "__main__":
    build_hetero_from_primekg()
