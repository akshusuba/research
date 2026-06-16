"""UniProt-accession -> HGNC-gene-symbol mapping (for cross-vocabulary metrics).

DrugMechDB curates mechanism-of-action paths whose protein nodes are identified
by UniProt accessions (e.g. ``UniProt:P00519``), while PrimeKG ``gene_protein``
nodes are HGNC gene symbols (e.g. ``ABL1``). To compare the two we need a
UniProt -> HGNC symbol map.

We resolve accessions through the public mygene.info batch REST API (no API key
or pip install required) and cache the result on disk at
``data/uniprot2symbol.json`` so re-runs are reproducible and work offline. Only
accessions missing from the cache trigger a network call.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List

from oncorepurpose.config import DATA_DIR

CACHE_PATH = DATA_DIR / "uniprot2symbol.json"
MYGENE_URL = "https://mygene.info/v3/query"
BATCH_SIZE = 800

# Official UniProt accession format (Swiss-Prot/TrEMBL). Used to drop obviously
# invalid tokens before hitting the network.
_ACC_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def _load_cache() -> Dict[str, str]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=0, sort_keys=True)
    tmp.replace(CACHE_PATH)


def _clean(acc: str) -> str:
    """Strip an optional ``UniProt:`` prefix and any isoform suffix (``-1``)."""
    acc = str(acc).strip()
    if acc.lower().startswith("uniprot:"):
        acc = acc.split(":", 1)[1]
    acc = acc.split("-", 1)[0]  # collapse isoforms P12345-2 -> P12345
    return acc.strip().upper()


def _query_mygene(accessions: List[str]) -> Dict[str, str]:
    """Resolve a list of accessions to symbols via mygene.info batch POST."""
    import requests

    out: Dict[str, str] = {}
    for i in range(0, len(accessions), BATCH_SIZE):
        chunk = accessions[i:i + BATCH_SIZE]
        resp = requests.post(
            MYGENE_URL,
            data={
                "q": ",".join(chunk),
                "scopes": "uniprot",
                "fields": "symbol",
                "species": "human",
            },
            timeout=60,
        )
        resp.raise_for_status()
        hits = resp.json()
        if not isinstance(hits, list):
            continue
        # A query can yield multiple hits; keep the first one carrying a symbol.
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            q = hit.get("query")
            sym = hit.get("symbol")
            if not q or not sym or hit.get("notfound"):
                continue
            q = str(q).upper()
            if q not in out:  # first (highest-scored) symbol wins
                out[q] = str(sym)
    return out


def uniprot_to_symbol(
    accessions: Iterable[str], use_network: bool = True
) -> Dict[str, str]:
    """Map UniProt accessions to HGNC gene symbols, with disk caching.

    Args:
        accessions: iterable of accessions, with or without ``UniProt:`` prefix.
        use_network: if False, only the on-disk cache is consulted.

    Returns:
        dict mapping cleaned accession -> HGNC symbol (only resolved entries).
    """
    cleaned = {_clean(a) for a in accessions}
    cleaned = {a for a in cleaned if a}

    cache = _load_cache()
    # Cache stores resolved symbols only; an accession present with empty string
    # means "queried, no symbol" so we don't re-query it forever.
    missing = sorted(a for a in cleaned if a not in cache and _ACC_RE.match(a))

    if missing and use_network:
        try:
            resolved = _query_mygene(missing)
        except Exception:
            resolved = {}
        for acc in missing:
            cache[acc] = resolved.get(acc, "")
        _save_cache(cache)

    return {a: cache[a] for a in cleaned if cache.get(a)}
