"""Best-effort open-access full-text fetch from Europe PMC.

Abstracts often omit the explicit mechanism-of-action sentence ("drug X inhibits
protein Y"), so when a retrieved record is in the PMC open-access subset we pull
its full text and harvest the gene-mentioning passages. This is a *bonus* signal:
the OA subset is small and most records 404, so every call degrades gracefully to
an empty result and the verifier proceeds on abstracts alone.

Europe PMC exposes OA full text at::

    https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{id}/fullTextXML

which returns JATS XML for the OA subset and 404 for everything else. We strip
tags, split into passages, and keep only passages that name a bridge gene. The
fetch is bounded (a couple of papers, short timeout) so it never dominates the
verify step's latency.
"""

from __future__ import annotations

import html
import re
from typing import Dict, List, Optional

import requests

FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{id}/fullTextXML"

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# Keep only the body-ish prose; drop the reference list which is noisy and huge.
_REFS = re.compile(r"<ref-list.*", re.DOTALL | re.IGNORECASE)


def _strip_xml(xml: str) -> str:
    """Turn JATS XML into plain prose: drop references, strip tags, unescape."""
    xml = _REFS.sub("", xml or "")
    text = _TAG.sub(" ", xml)
    return _WS.sub(" ", html.unescape(text)).strip()


def _gene_in_text(symbol: str, text: str) -> bool:
    if not symbol:
        return False
    return re.search(rf"\b{re.escape(symbol)}\b", text, re.IGNORECASE) is not None


def fetch_fulltext_record(source: str, rec_id: str, timeout: float = 8.0) -> Optional[str]:
    """Return stripped OA full text for one record, or None (404 / non-OA / error)."""
    if not source or not rec_id:
        return None
    url = FULLTEXT_URL.format(source=source, id=rec_id)
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200 or not r.text:
            return None
        return _strip_xml(r.text)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  [fulltext] fetch failed for {source}:{rec_id}: {exc}")
        return None


def fulltext_passages(
    records: List[Dict],
    gene_syms: List[str],
    max_papers: int = 2,
    timeout: float = 8.0,
    max_chars: int = 6000,
) -> List[Dict]:
    """Fetch OA full text for the top ``records`` and return gene-mentioning records.

    Each returned dict mirrors the ``search_literature`` abstract shape
    ({title, source, id, abstract, ...}) so it can be folded straight into the
    verifier's evidence pool. ``abstract`` holds the concatenated gene-mentioning
    passages (capped at ``max_chars``). Bounded to at most ``max_papers`` fetches
    with a short timeout; failures (the common case, since most papers are not OA)
    are silently skipped.
    """
    genes = [g for g in (gene_syms or []) if g]
    out: List[Dict] = []
    tried = 0
    for rec in records:
        if tried >= max_papers:
            break
        source, rec_id = rec.get("source", ""), rec.get("id", "")
        if not source or not rec_id:
            continue
        tried += 1
        text = fetch_fulltext_record(source, rec_id, timeout=timeout)
        if not text:
            continue
        passages = [
            p.strip() for p in re.split(r"(?<=[.!?])\s+", text)
            if any(_gene_in_text(g, p) for g in genes)
        ]
        if not passages:
            continue
        joined = " ".join(passages)[:max_chars]
        out.append({
            "title": rec.get("title", ""),
            "authors": rec.get("authors", ""),
            "year": rec.get("year", ""),
            "source": source,
            "id": rec_id,
            "abstract": joined,
            "fulltext": True,
        })
    return out
