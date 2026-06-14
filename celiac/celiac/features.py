"""
Text-embedding node features for inductive link prediction.

The central problem with the original setup is that every model used a learnable
``nn.Embedding`` keyed by node index. Such models can only score nodes that were
present at training time, and they win on transductive splits by memorising node
identity rather than using graph structure. To test whether the GNN provides real
value, every model must instead consume *content* features that are defined for
any node -- including nodes never seen during training.

This module turns each node's name/label (e.g. ``"Ataxia"``, ``"HLA-DQA1"``,
``"Bifidobacterium"``) into a fixed feature vector. The primary path uses a small
SentenceTransformer; if that is unavailable (no network / broken install) we fall
back to a deterministic character n-gram hashing embedding so the pipeline always
runs and remains reproducible.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_HASH_DIM = 256


def _node_texts(data: HeteroData, node_type: str) -> List[str]:
    """Return a human-readable string per node for the given type."""
    store = data[node_type]
    names = getattr(store, "node_names", None)
    num_nodes = int(store.num_nodes)

    if names is None:
        # No names available: fall back to a type-tagged index string so the
        # node still gets a (weakly informative) deterministic feature.
        return [f"{node_type} {i}" for i in range(num_nodes)]

    texts = []
    for i in range(num_nodes):
        raw = names[i] if i < len(names) else ""
        raw = "" if raw is None else str(raw).strip()
        # Prefix the node type so e.g. a gene and a phenotype with similar names
        # do not collapse to identical features.
        texts.append(f"{node_type.replace('_', ' ')}: {raw}" if raw else f"{node_type} {i}")
    return texts


def _hash_embed(texts: List[str], dim: int = DEFAULT_HASH_DIM) -> np.ndarray:
    """Deterministic character n-gram hashing embedding (offline fallback)."""
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        grams: List[str] = list(tokens)
        for tok in tokens:
            padded = f"#{tok}#"
            grams.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        for gram in grams:
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vectors[row, idx] += sign
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _sentence_transformer_embed(
    texts_by_type: Dict[str, List[str]],
    model_name: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Encode node texts with a SentenceTransformer; ``None`` on any failure."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - import/env dependent
        print(f"  [features] sentence-transformers unavailable ({exc}); using hashing fallback")
        return None

    try:
        model = SentenceTransformer(model_name)
        out: Dict[str, np.ndarray] = {}
        for node_type, texts in texts_by_type.items():
            emb = model.encode(
                texts,
                batch_size=256,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            out[node_type] = emb.astype(np.float32)
        return out
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        print(f"  [features] SentenceTransformer encoding failed ({exc}); using hashing fallback")
        return None


def build_text_features(
    data: HeteroData,
    cache_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
    force_fallback: bool = False,
) -> HeteroData:
    """Attach text-embedding features to ``data[node_type].x`` for every node type.

    Args:
        data: Heterogeneous graph with ``node_names`` per node store.
        cache_path: Optional ``.pt`` path to cache/reuse computed features.
        model_name: SentenceTransformer model id.
        use_cache: Reuse cached features if present and shapes match.
        force_fallback: Skip the transformer and use the hashing embedding
            (useful for fast, network-free smoke tests).

    Returns:
        The same ``data`` object with ``x`` populated (float32) per node type.
    """
    cache_path = Path(cache_path) if cache_path is not None else None

    if use_cache and cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        shapes_ok = all(
            nt in cached and cached[nt].shape[0] == int(data[nt].num_nodes)
            for nt in data.node_types
        )
        if shapes_ok:
            for nt in data.node_types:
                data[nt].x = cached[nt].float()
            dim = next(iter(cached.values())).shape[1]
            print(f"  [features] loaded cached text features (dim={dim}) from {cache_path.name}")
            return data
        print("  [features] cache shape mismatch; recomputing")

    texts_by_type = {nt: _node_texts(data, nt) for nt in data.node_types}

    emb_by_type: Optional[Dict[str, np.ndarray]] = None
    if not force_fallback:
        emb_by_type = _sentence_transformer_embed(texts_by_type, model_name)

    if emb_by_type is None:
        emb_by_type = {nt: _hash_embed(texts) for nt, texts in texts_by_type.items()}
        source = f"hashing (dim={DEFAULT_HASH_DIM})"
    else:
        dim = next(iter(emb_by_type.values())).shape[1]
        source = f"{model_name} (dim={dim})"

    tensors: Dict[str, torch.Tensor] = {}
    for nt in data.node_types:
        x = torch.from_numpy(np.ascontiguousarray(emb_by_type[nt])).float()
        data[nt].x = x
        tensors[nt] = x

    print(f"  [features] built node features via {source}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, cache_path)
        print(f"  [features] cached features to {cache_path.name}")

    return data


def feature_dims(data: HeteroData) -> Dict[str, int]:
    """Return the feature dimension per node type (must have ``x`` set)."""
    return {nt: int(data[nt].x.size(1)) for nt in data.node_types}
