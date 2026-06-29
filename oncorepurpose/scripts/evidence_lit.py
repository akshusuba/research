#!/usr/bin/env python
"""Shared literature/evidence utilities for OncoEvidence's load-bearing retrieval layer.

This module is used by both ``scripts/evidence_weighted_graph.py`` (evidence-weighted
mechanism scoring) and ``scripts/contradiction_detector.py`` (looking for evidence
AGAINST a claim). It provides:

  * A disk-backed Europe PMC cache so reruns are cheap and rate limits are
    respected (polite ~0.3s sleep between *live* calls only; cache hits never sleep).
  * ``count_recency`` -- a single cheap call returning (hitCount, latest_year) for a
    co-mention query, the raw material for co-mention strength + recency weighting.
  * A lexical contradiction/support sentence classifier (no LLM, no API key) that
    grades a retrieved sentence as supporting / contradicting / neutral for a claim
    that ``drug`` works in ``cancer``.
  * ``contradiction_scan`` -- issues contradiction-oriented Europe PMC queries for a
    (drug, cancer) pair, classifies the retrieved sentences, and returns a tally.

Network policy: every call is failure-tolerant. On any HTTP/network error the call
returns ``None`` (count) or an empty list (search) and the error is NOT cached, so a
later run with connectivity will fill it in. Definitive empty results ARE cached.
"""
from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from oncorepurpose.config import ONCOLOGY_KEYWORDS

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CURRENT_YEAR = datetime.now().year
_TAG = re.compile(r"<[^>]+>")
# Generic words inside a cancer name that are not specific enough to anchor relevance.
# Includes broad class nouns ('cancer', 'carcinoma', ...): anchoring on them would
# match a sentence about a DIFFERENT cancer of the same broad class (e.g. a tamoxifen
# 'breast cancer' resistance line should not count for a 'colorectal cancer' claim).
_CANCER_STOPWORDS = frozenset({
    "cell", "cells", "type", "disease", "malignant", "primary", "familial",
    "positive", "negative", "stage", "grade", "adult", "childhood", "chronic",
    "acute", "metastatic", "advanced", "recurrent", "refractory",
    "cancer", "cancers", "tumor", "tumors", "tumour", "tumours", "neoplasm",
    "neoplasms", "neoplasia", "carcinoma", "carcinomas", "sarcoma", "sarcomas",
    "malignancy", "malignancies", "syndrome", "tumoral",
})


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Strip Europe PMC HTML markup and unescape entities."""
    return html.unescape(_TAG.sub("", text or "")).strip()


def query_token(name: str) -> str:
    """Drop trailing parenthetical qualifiers (e.g. '(disease)', brand/salt) and quotes."""
    base = re.split(r"\s*\(", name or "")[0]
    return re.sub(r"\s+", " ", base.replace('"', " ")).strip()


def phrase(term: str) -> str:
    """Quote a term so Europe PMC treats it as an exact phrase."""
    return '"' + (term or "").replace('"', "").strip() + '"'


def mentions(term: str, text_low: str) -> bool:
    """Word-boundary, case-insensitive membership test against a lowercased text."""
    if not term:
        return False
    return re.search(rf"\b{re.escape(term.lower())}\b", text_low) is not None


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


# --------------------------------------------------------------------------- #
# Disk-backed Europe PMC cache
# --------------------------------------------------------------------------- #
class EPMCCache:
    """A polite, disk-backed Europe PMC client.

    Cache keys encode (query, page_size, want_abstracts, sort) so different shapes
    of the same query never collide. ``stats`` tracks cache hits / live calls /
    network errors for honest reporting.
    """

    def __init__(self, path: Path, sleep: float = 0.3, user_agent: str = "OncoEvidence-evidence/1.0"):
        self.path = Path(path)
        self.sleep = sleep
        self.cache: Dict[str, dict] = {}
        if self.path.exists():
            try:
                self.cache = json.loads(self.path.read_text())
            except Exception:
                self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.stats = {"hits": 0, "live": 0, "errors": 0, "empty": 0}
        self._since_save = 0

    def _key(self, query: str, page_size: int, want_abstracts: bool, sort: Optional[str]) -> str:
        return json.dumps([query, page_size, want_abstracts, sort], sort_keys=True)

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.cache, indent=0))
        tmp.replace(self.path)
        self._since_save = 0

    def query(self, query: str, page_size: int = 1, want_abstracts: bool = False,
              sort: Optional[str] = None) -> Optional[dict]:
        """Return {'hitCount': int, 'latest_year': int|None, 'results': [...]} or None.

        ``None`` means a network/HTTP error (not cached, retry-able). A definitive
        zero-hit response is a real (cached) result with hitCount 0.
        """
        key = self._key(query, page_size, want_abstracts, sort)
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]
        params = {"query": query, "format": "json", "pageSize": page_size,
                  "resultType": "core" if want_abstracts else "lite"}
        if sort:
            params["sort"] = sort
        try:
            r = self.session.get(EUROPE_PMC, params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
        except Exception as exc:  # network dependent -> do NOT cache
            self.stats["errors"] += 1
            if self.stats["errors"] <= 3:
                print(f"  [epmc] error on '{query[:60]}': {exc}")
            return None
        finally:
            self.stats["live"] = self.stats["live"]  # no-op; sleep handled below
        # polite delay only on a real live call
        time.sleep(self.sleep)
        self.stats["live"] += 1
        hit_count = int(j.get("hitCount", 0) or 0)
        raw_results = j.get("resultList", {}).get("result", []) or []
        results = []
        latest_year = None
        for it in raw_results:
            y = _parse_year(it)
            if y is not None:
                latest_year = y if latest_year is None else max(latest_year, y)
            results.append({
                "title": clean_text(it.get("title", "")),
                "year": y,
                "id": it.get("id", ""),
                "source": it.get("source", ""),
                "abstract": clean_text(it.get("abstractText", "")) if want_abstracts else "",
            })
        if hit_count == 0:
            self.stats["empty"] += 1
        out = {"hitCount": hit_count, "latest_year": latest_year, "results": results}
        self.cache[key] = out
        self._since_save += 1
        if self._since_save >= 25:
            self.save()
        return out


def _parse_year(rec: dict) -> Optional[int]:
    raw = (rec.get("firstPublicationDate", "") or "")[:4] or rec.get("pubYear")
    try:
        y = int(raw)
    except (TypeError, ValueError):
        return None
    if y < 1900 or y > CURRENT_YEAR:
        return None
    return y


# --------------------------------------------------------------------------- #
# Co-mention strength + recency (one call each)
# --------------------------------------------------------------------------- #
def loose_terms(name: str) -> str:
    """Cancer/long names as an implicit-AND of their words (no exact phrase).

    Europe PMC ANDs space-separated tokens, so a verbose ontology label like
    'malignant Sertoli-Leydig cell tumor of ovary' becomes a robust all-words match
    instead of a brittle exact phrase that returns zero hits. Short connective words
    are dropped to avoid over-constraining.
    """
    toks = [t for t in re.findall(r"[A-Za-z0-9-]+", query_token(name))
            if len(t) >= 3 and t.lower() not in {"the", "and", "for", "with", "type"}]
    return " AND ".join(toks)


def count_recency(cache: EPMCCache, term_a: str, term_b: Optional[str] = None,
                  phrase_b: bool = True) -> Tuple[Optional[int], Optional[int]]:
    """(hitCount, latest_year) for a co-mention query.

    A single Europe PMC call sorted by first-publication-date descending gives us
    both the total co-mention count and the most recent co-mention year. ``term_a``
    is always exact-phrased (drug or gene symbol -- precise); ``term_b`` is exact-
    phrased when ``phrase_b`` is True (gene symbol) or matched as an implicit-AND of
    its words when False (a verbose cancer label). Returns (None, None) on network
    error so the caller can fall back to cache / skip.
    """
    a = query_token(term_a)
    if not a:
        return 0, None
    if term_b is None:
        q = phrase(a)
    else:
        if phrase_b:
            b = query_token(term_b)
            if not b:
                return 0, None
            q = f"{phrase(a)} AND {phrase(b)}"
        else:
            b = loose_terms(term_b)
            if not b:
                return 0, None
            q = f"{phrase(a)} AND ({b})"
    res = cache.query(q, page_size=1, want_abstracts=False, sort="FIRST_PDATE_D desc")
    if res is None:
        return None, None
    return res["hitCount"], res["latest_year"]


def recency_weight(year: Optional[int], tau: float = 8.0, floor: float = 0.2) -> float:
    """Exponential recency in (floor, 1]; older / missing -> closer to floor."""
    if year is None:
        return floor
    age = max(0, CURRENT_YEAR - year)
    import math
    return floor + (1.0 - floor) * math.exp(-age / tau)


# --------------------------------------------------------------------------- #
# Lexical contradiction / support classifier (no LLM)
# --------------------------------------------------------------------------- #
# Phrases that indicate a drug did NOT help / failed / lost efficacy in a cancer.
_CONTRA_PATTERNS = [
    r"no (?:significant |clinical |survival |overall )?benefit",
    r"did not (?:improve|prolong|increase|reduce|meet|show|demonstrate)",
    r"no improvement", r"failed to (?:improve|demonstrate|meet|show|prolong)",
    r"lack of (?:efficacy|benefit|response|activity)",
    r"\bineffective\b", r"\bnot effective\b", r"no(?:t a)? significant difference",
    r"no(?:t a)? statistically significant", r"did not differ",
    r"negative (?:trial|study|result)", r"not superior", r"no survival benefit",
    r"no(?:t)? (?:objective )?response", r"disease progression", r"\bprogressed\b",
    r"\brefractory\b", r"\bresistance\b", r"\bresistant\b", r"\bunresponsive\b",
    r"no clinical activity", r"limited efficacy", r"poor response",
    r"\brelapse", r"treatment failure", r"acquired resistance",
    r"insensitiv", r"loss of (?:response|sensitivity)",
]
# Phrases that indicate the drug helped / was active in a cancer.
_SUPPORT_PATTERNS = [
    r"\bimproved (?:survival|outcome|response|os|pfs)",
    r"prolonged survival", r"significant(?:ly)? (?:improv|increas|prolong|reduc)",
    r"objective response", r"response rate", r"\befficacious\b", r"\befficacy\b",
    r"clinical benefit", r"durable response", r"complete response",
    r"partial response", r"tumou?r regression", r"inhibit(?:ed|s)? (?:tumou?r |cell )?growth",
    r"suppress(?:ed|es)? (?:tumou?r|growth|proliferation)", r"reduced tumou?r",
    r"\bactive against\b", r"antitumou?r activity", r"promising",
    r"survival benefit", r"well tolerated", r"sensiti[sz]e",
    r"effective (?:treatment|therapy|against|in)",
]
# Simple negation guard: "no resistance", "without resistance", "overcome resistance"
# should NOT count as a contradiction.
_NEG_GUARD = re.compile(
    r"\b(no|without|overcome|overcoming|overcame|reverse[sd]?|reversing|"
    r"prevent(?:s|ed|ing)?|delay(?:s|ed|ing)?|circumvent\w*|avoid\w*)\b\s+\w{0,12}\s*$",
)

_CONTRA_RE = [re.compile(p, re.IGNORECASE) for p in _CONTRA_PATTERNS]
_SUPPORT_RE = [re.compile(p, re.IGNORECASE) for p in _SUPPORT_PATTERNS]


def _find_cues(sentence: str, patterns) -> List[str]:
    hits = []
    for rx in patterns:
        m = rx.search(sentence)
        if not m:
            continue
        # negation guard: ignore a contradiction cue immediately preceded by a
        # cancelling word (e.g. "overcome resistance", "no resistance").
        prefix = sentence[max(0, m.start() - 30):m.start()]
        if _NEG_GUARD.search(prefix.lower() + " "):
            continue
        hits.append(m.group(0).lower())
    return hits


def classify_sentence(sentence: str) -> Dict:
    """Grade a sentence as supporting / contradicting / neutral with the cue lists.

    Returns {'label', 'contra_cues', 'support_cues'}. When both kinds of cue are
    present the sentence is graded 'mixed' (counted toward neither tally) so noisy
    both-ways sentences do not inflate either side.
    """
    contra = _find_cues(sentence, _CONTRA_RE)
    support = _find_cues(sentence, _SUPPORT_RE)
    if contra and support:
        label = "mixed"
    elif contra:
        label = "contradicting"
    elif support:
        label = "supporting"
    else:
        label = "neutral"
    return {"label": label, "contra_cues": sorted(set(contra)),
            "support_cues": sorted(set(support))}


def contradiction_queries(drug: str, cancer: str) -> List[str]:
    """Contradiction-oriented Europe PMC queries for a (drug, cancer) claim.

    The drug is exact-phrased; the cancer is matched as an implicit-AND of its words
    (verbose ontology labels rarely appear verbatim), so we do not miss the negative
    literature for cancers with long names.
    """
    d = query_token(drug)
    c_loose = loose_terms(cancer)
    if not d or not c_loose:
        return []
    base = f"{phrase(d)} AND ({c_loose})"
    return [
        f"{base} AND (resistance OR resistant OR refractory)",
        f"{base} AND (ineffective OR \"no benefit\" OR \"did not improve\" OR "
        f"\"lack of efficacy\" OR \"no response\")",
        f"{base} AND (\"failed trial\" OR \"negative trial\" OR \"not superior\" OR "
        f"\"no survival benefit\" OR \"treatment failure\")",
    ]


def _cancer_anchors(cancer_tok: str) -> List[str]:
    """Specific words inside a cancer name usable to anchor sentence relevance."""
    words = re.findall(r"[a-z]+", cancer_tok.lower())
    return [w for w in words if len(w) >= 5 and w not in _CANCER_STOPWORDS]


def _cancer_relevant(low: str, cancer_tok: str, anchors: List[str]) -> bool:
    """True if a sentence is plausibly about THIS specific cancer.

    Requires the sentence to name the full cancer phrase or one of its specific
    words (e.g. 'glioblastoma', or 'lymphoma' from 'non-Hodgkin lymphoma'). We do
    NOT accept a generic oncology keyword: a tamoxifen+glioblastoma document can
    contain a sentence about tamoxifen resistance in *breast* cancer, and that
    sentence must not be counted against the glioblastoma claim. Conservative (it
    misses abbreviation-only references like 'GBM'), but precise.
    """
    if cancer_tok and mentions(cancer_tok, low):
        return True
    if anchors:
        return any(mentions(w, low) for w in anchors)
    # No usable anchor word (very short/generic cancer name): fall back to a generic
    # oncology keyword so we are not left with zero relevance signal.
    return any(k in low for k in ONCOLOGY_KEYWORDS)


def contradiction_scan(cache: EPMCCache, drug: str, cancer: str,
                       per_query: int = 12, max_sentences: int = 40) -> Dict:
    """Query contradiction phrasings and tally supporting/contradicting/neutral sentences.

    A sentence is classified when it mentions the drug AND carries a cancer signal
    (full phrase, a specific cancer word, or an oncology keyword). The retrieved
    documents are already drug+cancer co-mentions, so this anchors relevance without
    demanding the verbatim ontology label (which rarely appears in a sentence that
    also names the drug). Returns a tally plus example sentences.
    """
    d_tok, c_tok = query_token(drug).lower(), query_token(cancer).lower()
    anchors = _cancer_anchors(c_tok)
    queries = contradiction_queries(drug, cancer)
    seen_ids: set = set()
    sentences: List[Dict] = []
    tally = {"supporting": 0, "contradicting": 0, "neutral": 0, "mixed": 0}
    n_abstracts = 0
    network_error = False

    for q in queries:
        res = cache.query(q, page_size=per_query, want_abstracts=True)
        if res is None:
            network_error = True
            continue
        for rec in res["results"]:
            rid = rec.get("id") or rec.get("title", "")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            text = f"{rec.get('title','')}. {rec.get('abstract','')}"
            if not rec.get("abstract"):
                continue
            n_abstracts += 1
            for sent in split_sentences(text):
                low = sent.lower()
                if not mentions(d_tok, low):
                    continue
                if not _cancer_relevant(low, c_tok, anchors):
                    continue
                cls = classify_sentence(sent)
                tally[cls["label"]] += 1
                if cls["label"] in ("contradicting", "supporting") and len(sentences) < max_sentences:
                    sentences.append({
                        "label": cls["label"], "sentence": sent[:400],
                        "cues": cls["contra_cues"] + cls["support_cues"],
                        "source": rec.get("source", ""), "id": rec.get("id", ""),
                        "year": rec.get("year"),
                    })

    support_n = tally["supporting"]
    contra_n = tally["contradicting"]
    total_signed = support_n + contra_n
    contra_fraction = (contra_n / total_signed) if total_signed else 0.0
    return {
        "drug": drug, "cancer": query_token(cancer),
        "n_abstracts": n_abstracts,
        "tally": tally,
        "supporting": support_n, "contradicting": contra_n,
        "contra_fraction": round(contra_fraction, 3),
        "sentences": sentences,
        "network_error": network_error,
    }
