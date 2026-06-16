"""Mechanism verification: is a proposed MOA path supported by the literature?

This is OncoEvidence's evidence-reviewer step (Aim 3). Given a multi-hop
mechanism path (drug -> target -> ... -> cancer gene -> cancer), we retrieve
abstracts and decide whether the literature supports that mechanism. Two modes:

* ``llm`` (when ``ONCO_LLM_API_KEY`` is set): the model reads the abstracts and
  returns a grade (supported / weak / contradicted / unknown) with a verbatim
  evidence quote, using ONLY the retrieved text.
* ``lexical`` (always available, no key/cost): checks whether the path's drug,
  bridge gene(s), and a mechanistic cue co-occur in the retrieved abstracts.

The lexical mode lets the whole pipeline run and be evaluated offline; the LLM
mode upgrades the grounding when an API key is available.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from oncorepurpose.agent.evidence_report import search_literature
from oncorepurpose.agent.llm import chat, llm_available

GRADES = ("supported", "weak", "contradicted", "unknown")
_CUES = ("inhibit", "inhibitor", "target", "targets", "treatment", "treat",
         "therapy", "therapeutic", "suppress", "block", "antagonist", "agonist",
         "mechanism", "mutation", "overexpress", "activate")


def _drug_token(name: str) -> str:
    return re.split(r"\s*\(", name)[0].strip().lower()


def _mentions(term: str, text: str) -> bool:
    """Word-boundary match (gene symbols are short and collision-prone)."""
    if not term:
        return False
    return re.search(rf"\b{re.escape(term.lower())}\b", text) is not None


def lexical_grade(path: Dict, abstracts: List[Dict]) -> Dict:
    """Heuristic grounding from co-occurrence of drug + gene + mechanistic cue."""
    corpus = " ".join(
        f"{a.get('title','')} {a.get('abstract','')}" for a in abstracts
    ).lower()
    if not corpus.strip():
        return {"grade": "unknown", "evidence": "no literature retrieved"}

    genes = [g for g in path.get("genes", []) if g]
    drug = _drug_token(path.get("drug", ""))
    gene_hit = any(_mentions(g, corpus) for g in genes)
    drug_hit = bool(drug) and drug in corpus
    cue_hit = any(c in corpus for c in _CUES)

    hit_genes = [g for g in genes if _mentions(g, corpus)]
    if gene_hit and drug_hit and cue_hit:
        grade = "supported"
    elif gene_hit and (drug_hit or cue_hit):
        grade = "weak"
    elif gene_hit or drug_hit:
        grade = "weak"
    else:
        grade = "unknown"
    ev = (f"co-mentions: drug={drug_hit}, genes={hit_genes or None}, "
          f"mechanistic_cue={cue_hit}")
    return {"grade": grade, "evidence": ev}


def _llm_prompt(path: Dict, abstracts: List[Dict]) -> List[Dict]:
    lit = "\n\n".join(
        f"[{a['source']}:{a['id']}] {a['title']}\n{a['abstract'][:1200]}"
        for a in abstracts if a.get("abstract")
    ) or "(no abstracts retrieved)"
    sys = (
        "You are a strict biomedical evidence reviewer. You are given a proposed "
        "drug mechanism-of-action path and a set of paper abstracts. Decide whether "
        "the abstracts SUPPORT the mechanism. Use ONLY the provided abstracts; do "
        "not use outside knowledge or invent citations. Respond ONLY as JSON with "
        "keys: grade (one of supported|weak|contradicted|unknown), evidence_quote "
        "(<=40-word verbatim quote from an abstract, or empty), rationale (<=30 words)."
    )
    usr = f"Mechanism path:\n{path['text']}\n\nAbstracts:\n{lit}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def _parse_json(text: Optional[str]) -> Dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


def llm_grade(path: Dict, abstracts: List[Dict]) -> Optional[Dict]:
    if not llm_available():
        return None
    res = _parse_json(chat(_llm_prompt(path, abstracts), json_mode=True, temperature=0.0))
    if not res or res.get("grade") not in GRADES:
        return None
    return {"grade": res["grade"], "evidence": res.get("evidence_quote", ""),
            "rationale": res.get("rationale", "")}


def verify_mechanism(path: Dict, n_lit: int = 5, use_llm: bool = True) -> Dict:
    """Retrieve abstracts for the path and grade mechanistic support.

    Query couples the drug with its primary bridge gene so retrieval targets the
    *mechanism*, not just the indication.
    """
    drug = _drug_token(path.get("drug", ""))
    genes = path.get("genes", [])
    query = f'{drug} AND {genes[0]}' if genes else drug
    abstracts = search_literature(query, page_size=n_lit, with_abstract=True)
    lex = lexical_grade(path, abstracts)
    out = {
        "path": path["text"], "type": path.get("type"),
        "query": query, "n_abstracts": len([a for a in abstracts if a.get("abstract")]),
        "lexical": lex, "llm": None,
        "grade": lex["grade"], "source": "lexical",
        "citations": [f"{a['source']}:{a['id']}" for a in abstracts if a.get("id")][:n_lit],
    }
    if use_llm:
        llm = llm_grade(path, abstracts)
        if llm is not None:
            out["llm"] = llm
            out["grade"] = llm["grade"]
            out["source"] = "llm"
    return out
