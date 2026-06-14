"""Shared text-embedding node features for PrimeKG.

The same features are fed to BOTH the GNN and the XGBoost baseline, so any
performance gap is attributable to graph topology rather than node content.
Each node is embedded from its name (and clinical description if available)
with a SentenceTransformer; a deterministic hashing fallback keeps the pipeline
runnable offline / for fast smoke tests.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from oncorepurpose.config import FEATURE_CACHE, TEXT_MODEL

DEFAULT_HASH_DIM = 256


def _node_texts(data: HeteroData, node_type: str) -> List[str]:
    store = data[node_type]
    names = getattr(store, "node_names", None)
    n = int(store.num_nodes)
    pretty = node_type.replace("_", " ")
    if names is None:
        return [f"{pretty} {i}" for i in range(n)]
    out = []
    for i in range(n):
        raw = names[i] if i < len(names) else ""
        raw = "" if raw is None else str(raw).strip()
        out.append(f"{pretty}: {raw}" if raw else f"{pretty} {i}")
    return out


def _hash_embed(texts: List[str], dim: int = DEFAULT_HASH_DIM) -> np.ndarray:
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        toks = re.findall(r"[a-z0-9]+", text.lower())
        grams: List[str] = list(toks)
        for tok in toks:
            p = f"#{tok}#"
            grams.extend(p[i : i + 3] for i in range(len(p) - 2))
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vecs[row, h % dim] += 1.0 if (h // dim) % 2 == 0 else -1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _st_embed(texts_by_type: Dict[str, List[str]], model_name: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        print(f"  [features] sentence-transformers unavailable ({exc}); hashing fallback")
        return None
    try:
        model = SentenceTransformer(model_name)
        out: Dict[str, np.ndarray] = {}
        for nt, texts in texts_by_type.items():
            out[nt] = model.encode(
                texts, batch_size=512, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True,
            ).astype(np.float32)
        return out
    except Exception as exc:  # pragma: no cover
        print(f"  [features] SentenceTransformer failed ({exc}); hashing fallback")
        return None


def build_text_features(
    data: HeteroData,
    cache_path: Optional[Path] = FEATURE_CACHE,
    model_name: str = TEXT_MODEL,
    force_fallback: bool = False,
) -> HeteroData:
    """Attach `data[type].x` text features for every node type."""
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and force_fallback:
        cache_path = cache_path.with_name(cache_path.stem + "_hash" + cache_path.suffix)

    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        if all(nt in cached and cached[nt].shape[0] == int(data[nt].num_nodes) for nt in data.node_types):
            for nt in data.node_types:
                data[nt].x = cached[nt].float()
            print(f"  [features] loaded cached features (dim={next(iter(cached.values())).shape[1]})")
            return data

    texts = {nt: _node_texts(data, nt) for nt in data.node_types}
    emb = None if force_fallback else _st_embed(texts, model_name)
    if emb is None:
        emb = {nt: _hash_embed(t) for nt, t in texts.items()}
        src = f"hashing (dim={DEFAULT_HASH_DIM})"
    else:
        src = f"{model_name} (dim={next(iter(emb.values())).shape[1]})"

    tensors = {}
    for nt in data.node_types:
        x = torch.from_numpy(np.ascontiguousarray(emb[nt])).float()
        data[nt].x = x
        tensors[nt] = x
    print(f"  [features] built node features via {src}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, cache_path)
        print(f"  [features] cached to {cache_path.name}")
    return data
