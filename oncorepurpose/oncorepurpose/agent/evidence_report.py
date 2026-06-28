"""Agentic evidence-report layer over top GNN repurposing predictions.

For each predicted (drug, cancer) candidate we:
1. retrieve literature via Europe PMC (no API key required),
2. assemble the GNN multi-hop KG rationale + literature into a RAG prompt,
3. ask an LLM to write a structured evidence dossier, and
4. run an LLM-as-judge pass that scores biological plausibility / evidence
   strength (1-5) to triage and rank candidates.

If no LLM API key is configured, steps 3-4 are skipped gracefully and the report
still contains the model score, KG rationale, and retrieved literature.
"""

from __future__ import annotations

import html
import json
import re
from typing import Dict, List, Optional

import requests

from oncorepurpose.agent.llm import chat, llm_available

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_TAG = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    """Strip HTML tags (Europe PMC marks up gene/abstract text) and unescape."""
    return html.unescape(_TAG.sub("", text or "")).strip()


def search_literature(query: str, page_size: int = 5, with_abstract: bool = True) -> List[Dict]:
    """Return [{title, authors, year, source, id, abstract}] from Europe PMC.

    ``with_abstract`` uses resultType=core to retrieve the abstract text, so the
    verifier can ground on what the paper actually says rather than just titles.
    """
    result_type = "core" if with_abstract else "lite"
    try:
        r = requests.get(
            EUROPE_PMC,
            params={"query": query, "format": "json", "pageSize": page_size,
                    "resultType": result_type},
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("resultList", {}).get("result", [])
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  [lit] Europe PMC failed: {exc}")
        return []
    out = []
    for it in results:
        out.append({
            "title": _clean(it.get("title", "")),
            "authors": it.get("authorString", ""),
            "year": it.get("pubYear", ""),
            "source": it.get("source", ""),
            "id": it.get("id", ""),
            "abstract": _clean(it.get("abstractText", "")) if with_abstract else "",
        })
    return out


def _dossier_prompt(drug: str, disease: str, score: float, paths: List[Dict],
                    lit: List[Dict], abstract_chars: int = 600) -> List[Dict]:
    path_txt = "\n".join(f"- {p['text']}" for p in paths) or "- (no short KG path found)"

    def _ref_block(l: Dict) -> str:
        head = f"- [{l['source']}:{l['id']}] {l['title']} ({l['year']})"
        abs = (l.get("abstract") or "").strip()
        if abs:
            abs = abs[:abstract_chars].rsplit(" ", 1)[0] + ("..." if len(abs) > abstract_chars else "")
            return f"{head}\n    abstract: {abs}"
        return f"{head}\n    abstract: (none available)"

    lit_txt = "\n".join(_ref_block(l) for l in lit) or "- (no literature retrieved)"
    sys = (
        "You are a cautious biomedical research assistant. Using ONLY the provided "
        "knowledge-graph rationale and retrieved literature (title + abstract), write a "
        "concise evidence dossier for a proposed drug-repurposing hypothesis. Ground every "
        "claim in the supplied abstracts and do not invent citations or facts not present "
        "in them. Clearly separate mechanistic rationale, supporting evidence, "
        "contradicting/uncertain evidence, and a one-line verdict. This is "
        "hypothesis-generating, not medical advice."
    )
    usr = (
        f"Hypothesis: repurpose '{drug}' for '{disease}'.\n"
        f"Model confidence score: {score:.3f}\n\n"
        f"Knowledge-graph multi-hop rationale:\n{path_txt}\n\n"
        f"Retrieved literature (title + abstract):\n{lit_txt}\n\n"
        "Write the dossier in markdown with sections: Mechanistic rationale, "
        "Supporting evidence, Contradicting/uncertain, Verdict. Cite the [source:id] tag "
        "next to each evidence statement you draw from an abstract."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def _judge_prompt(drug: str, disease: str, dossier: str) -> List[Dict]:
    sys = (
        "You are a strict scientific reviewer acting as an LLM-as-judge. Given an "
        "evidence dossier for a drug-repurposing hypothesis, rate it. Respond ONLY as "
        "JSON with keys: plausibility (1-5 int), evidence_strength (1-5 int), "
        "novelty (1-5 int), rationale (string, <=40 words), recommend (boolean)."
    )
    usr = f"Drug: {drug}\nDisease: {disease}\n\nDossier:\n{dossier}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def _parse_judge(text: Optional[str]) -> Dict:
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


def build_candidate_report(
    drug_name: str, disease_name: str, score: float, paths: List[Dict],
    n_lit: int = 5, use_llm: bool = True,
) -> Dict:
    """Produce a single candidate's evidence report (RAG dossier + judge score)."""
    lit = search_literature(f'{drug_name} AND {disease_name}', page_size=n_lit)
    report = {
        "drug": drug_name, "disease": disease_name, "model_score": score,
        "kg_paths": paths, "literature": lit,
        "dossier": None, "judge": {},
    }
    if use_llm and llm_available():
        dossier = chat(_dossier_prompt(drug_name, disease_name, score, paths, lit))
        report["dossier"] = dossier
        if dossier:
            report["judge"] = _parse_judge(
                chat(_judge_prompt(drug_name, disease_name, dossier), json_mode=True, temperature=0.0)
            )
    return report


def rank_reports(reports: List[Dict]) -> List[Dict]:
    """Rank candidates by judge plausibility+evidence (if present), else model score."""
    def keyf(r):
        j = r.get("judge") or {}
        # Tie-break / fall back on disease-specific lift, not raw score, to avoid
        # re-introducing the popularity ordering.
        spec = float(r.get("specificity_lift", r["model_score"]))
        if j:
            return (float(j.get("plausibility", 0)) + float(j.get("evidence_strength", 0)), spec)
        return (-1.0, spec)
    return sorted(reports, key=keyf, reverse=True)
