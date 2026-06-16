"""Mechanism-focused literature retrieval over Europe PMC.

The single ``drug AND gene`` query used previously frequently misses the
canonical mechanism-of-action (MOA) abstract, so the downstream verifier returns
"unknown" too often. ``retrieve_for_mechanism`` instead issues a small *set* of
complementary Europe PMC queries (an exact-phrase pairing, a mechanism-cued
pairing, and -- when available -- an indication query) and merges the results,
de-duplicating by record id while prioritising records that actually carry an
abstract. This surfaces the MOA paper far more reliably for the same cost of a
handful of polite requests.

The merged records keep the exact same dict shape returned by
``search_literature`` ({title, authors, year, source, id, abstract}), so the
verifier and any other consumer can use them interchangeably.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from oncorepurpose.agent.evidence_report import search_literature

# Mechanistic terms OR-ed into one query variant so retrieval is biased toward
# papers that describe *what the drug does to the target*, not just co-mention.
_MECH_TERMS = (
    "mechanism OR inhibitor OR inhibits OR inhibition OR target OR targets "
    "OR agonist OR antagonist OR activates OR activation OR binds OR modulates"
)


def _drug_token(name: str) -> str:
    """Drop trailing parenthetical qualifiers (e.g. brand/salt) for cleaner queries."""
    return re.split(r"\s*\(", name or "")[0].strip()


def _phrase(term: str) -> str:
    """Quote a term so Europe PMC treats it as an exact phrase."""
    return '"' + term.replace('"', "").strip() + '"'


def _gene_in_text(symbol: str, rec: Dict) -> bool:
    """Word-boundary, case-insensitive check that a gene symbol is in title/abstract."""
    if not symbol:
        return False
    text = f"{rec.get('title', '')} {rec.get('abstract', '')}"
    return re.search(rf"\b{re.escape(symbol)}\b", text, re.IGNORECASE) is not None


def _build_queries(drug: str, genes: List[str], disease: Optional[str]) -> List[str]:
    """Complementary query variants, most mechanism-specific first."""
    drug = _drug_token(drug)
    genes = [g for g in (genes or []) if g]
    queries: List[str] = []
    if drug and genes:
        primary = genes[0]
        # Exact-phrase pairing: precise co-mention of drug + bridge gene.
        queries.append(f"{_phrase(drug)} AND {_phrase(primary)}")
        # Mechanism-cued pairing: bias toward MOA statements on that target.
        queries.append(f"{_phrase(drug)} AND {_phrase(primary)} AND ({_MECH_TERMS})")
        # Field-targeted: force the gene to appear in the title/abstract (not just
        # full text), so the abstracts we score actually name the bridge gene.
        queries.append(
            f"{_phrase(drug)} AND (ABSTRACT:{_phrase(primary)} OR TITLE:{_phrase(primary)})"
        )
        # Second bridge gene (PPI / pathway paths) if present.
        if len(genes) > 1:
            queries.append(f"{_phrase(drug)} AND {_phrase(genes[1])}")
    elif drug:
        queries.append(_phrase(drug))
    # Indication query catches papers that frame the drug for the disease.
    if drug and disease:
        queries.append(f"{_phrase(drug)} AND {_phrase(_drug_token(disease))}")
    return queries


def retrieve_for_mechanism(
    drug: str,
    genes: List[str],
    disease: Optional[str] = None,
    n: int = 6,
    per_query: int = 6,
) -> List[Dict]:
    """Retrieve & merge mechanism-relevant abstracts for a drug/gene(s) pair.

    Issues several complementary Europe PMC queries (exact-phrase drug+gene, a
    mechanism-cued variant, an optional second gene, and an optional indication
    query), then merges them de-duplicating by record id. Records that carry an
    abstract are preferred and placed first, since the verifier grounds on
    abstract text. Robust to network errors: each underlying call already returns
    ``[]`` on failure, so a partial outage just yields fewer (or no) records.

    Returns up to ``n`` dicts with the same keys as ``search_literature``.
    """
    queries = _build_queries(drug, genes, disease)
    if not queries:
        return []
    gene_syms = [g for g in (genes or []) if g]

    seen: "dict[str, Dict]" = {}
    order: List[str] = []
    # Preserve query priority order: earlier (more specific) queries win the slot,
    # but a later hit that carries an abstract upgrades an earlier abstract-less one.
    for q in queries:
        for rec in search_literature(q, page_size=per_query, with_abstract=True):
            rid = rec.get("id") or ""
            key = rid or (rec.get("source", "") + "|" + rec.get("title", ""))
            if not key:
                continue
            if key not in seen:
                seen[key] = rec
                order.append(key)
            elif rec.get("abstract") and not seen[key].get("abstract"):
                seen[key] = rec  # upgrade to the version that has an abstract

    # Rank for the verifier: abstracts that actually name a bridge gene first,
    # then any abstract, then abstract-less stubs -- stable within each tier.
    def tier(key: str) -> int:
        rec = seen[key]
        has_abs = bool(rec.get("abstract"))
        names_gene = any(_gene_in_text(g, rec) for g in gene_syms)
        if has_abs and names_gene:
            return 0
        if has_abs:
            return 1
        return 2

    ranked = sorted(enumerate(order), key=lambda ik: (tier(ik[1]), ik[0]))
    return [seen[k] for _, k in ranked[:n]]
