"""
Dataset loading for the inductive PrimeKG-Celiac experiment.

Loads the cached PrimeKG-Celiac subgraph, attaches shared text-embedding node
features, and identifies the target edge type for link prediction.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from celiac.config import MODELS_DIR, PROJECT_ROOT
from celiac.features import build_text_features

EdgeType = Tuple[str, str, str]

# Default cached PrimeKG-Celiac heterograph and the target relation
# (disease -> phenotype, the 2,104-edge relation used in prior results).
DEFAULT_PRIMEKG_PT = PROJECT_ROOT / "data" / "primekg" / "celiac_heterodata.pt"
TARGET_EDGE_TYPE: EdgeType = ("disease", "disease_phenotype_positive", "effect_phenotype")
FEATURE_CACHE = MODELS_DIR / "primekg_text_features.pt"


def detect_target_edge_type(data: HeteroData) -> EdgeType:
    """Pick the disease->phenotype relation, else the largest cross-type relation."""
    if TARGET_EDGE_TYPE in data.edge_types:
        return TARGET_EDGE_TYPE

    # Fallback: largest edge type connecting two *different* node types.
    best: Optional[EdgeType] = None
    best_count = -1
    for et in data.edge_types:
        src, _, dst = et
        if src == dst:
            continue
        count = data[et].edge_index.size(1)
        if count > best_count:
            best_count = count
            best = et
    if best is None:
        raise ValueError("No cross-type edge found to use as a target.")
    return best


def load_primekg(
    pt_path: Optional[Path] = None,
    with_features: bool = True,
    feature_cache: Optional[Path] = FEATURE_CACHE,
    force_fallback_features: bool = False,
) -> Tuple[HeteroData, EdgeType]:
    """Load the PrimeKG-Celiac heterograph with text features.

    Returns:
        (data, target_edge_type)
    """
    pt_path = Path(pt_path) if pt_path is not None else DEFAULT_PRIMEKG_PT
    if not pt_path.exists():
        raise FileNotFoundError(
            f"PrimeKG heterograph not found at {pt_path}. "
            "Run the PrimeKG subgraph extraction first."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data: HeteroData = torch.load(pt_path, weights_only=False)

    # Ensure every node store exposes num_nodes for downstream code.
    for nt in data.node_types:
        if not hasattr(data[nt], "num_nodes") or data[nt].num_nodes is None:
            names = getattr(data[nt], "node_names", None)
            data[nt].num_nodes = len(names) if names is not None else 0

    if with_features:
        # Keep fallback (hashing) and transformer features in separate caches so a
        # fast smoke-test run never poisons the real-feature cache.
        cache_path = feature_cache
        if cache_path is not None and force_fallback_features:
            cache_path = cache_path.with_name(cache_path.stem + "_hash" + cache_path.suffix)
        build_text_features(
            data,
            cache_path=cache_path,
            use_cache=True,
            force_fallback=force_fallback_features,
        )

    target_edge_type = detect_target_edge_type(data)
    return data, target_edge_type


def graph_summary(data: HeteroData, target_edge_type: EdgeType) -> str:
    """Human-readable one-paragraph summary of the loaded graph."""
    lines = ["PrimeKG-Celiac graph:"]
    total_nodes = 0
    for nt in data.node_types:
        n = int(data[nt].num_nodes)
        total_nodes += n
        dim = int(data[nt].x.size(1)) if "x" in data[nt] else None
        lines.append(f"  node {nt}: {n}" + (f" (feat dim {dim})" if dim else ""))
    total_edges = sum(data[et].edge_index.size(1) for et in data.edge_types)
    lines.append(f"  total nodes: {total_nodes}, total edges: {total_edges}")
    lines.append(f"  target edge: {target_edge_type} = {data[target_edge_type].edge_index.size(1)} edges")
    return "\n".join(lines)
