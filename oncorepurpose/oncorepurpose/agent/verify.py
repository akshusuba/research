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

from oncorepurpose.agent.fulltext import fulltext_passages
from oncorepurpose.agent.llm import chat, llm_available
from oncorepurpose.agent.retrieval import retrieve_for_mechanism

GRADES = ("supported", "weak", "contradicted", "unknown")
_CUES = ("inhibit", "inhibitor", "target", "targets", "treatment", "treat",
         "therapy", "therapeutic", "suppress", "block", "antagonist", "agonist",
         "mechanism", "mutation", "overexpress", "activate")

# Mechanism-of-action cues: the drug *acts on* the named protein/gene. Presence of
# one of these in a drug+gene sentence is what distinguishes genuine MOA evidence
# from a mere statistical/pharmacogenetic co-mention.
_MOA_CUE = re.compile(
    r"\b(inhibit\w*|bind\w*|target\w*|substrate\w*|activat\w*|antagoni\w*|agoni\w*|"
    r"block\w*|suppress\w*|degrad\w*|modulat\w*|abrogat\w*|disrupt\w*|"
    r"phosphorylat\w*|stabili[sz]\w*|sequester\w*|interact\w*)\b",
    re.IGNORECASE,
)
# Pharmacogenetic / statistical-association language: a SNP, genotype, polymorphism,
# resistance, or prognosis association is NOT mechanism-of-action support.
_ASSOC_CUE = re.compile(
    r"\b(rs\d+|polymorphism\w*|genotyp\w*|snp|snps|variant\w*|allele\w*|gwas|"
    r"haplotype\w*|prognos\w*|surviv\w*|associat\w*|susceptibilit\w*|"
    r"predispos\w*|risk allele)\b",
    re.IGNORECASE,
)


def _drug_token(name: str) -> str:
    return re.split(r"\s*\(", name)[0].strip().lower()


def _mentions(term: str, text: str) -> bool:
    """Word-boundary match (gene symbols are short and collision-prone)."""
    if not term:
        return False
    return re.search(rf"\b{re.escape(term.lower())}\b", text) is not None


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (abstracts have no markup at this point)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def mechanism_sentences(path: Dict, abstracts: List[Dict]) -> List[Dict]:
    """Sentences from the evidence that name BOTH the drug and a path gene.

    Sentence-level grounding: rather than letting the LLM roam a whole abstract
    (where a tangential pharmacogenetic line can be mistaken for mechanism), we
    isolate the sentences that co-mention the drug token and at least one bridge
    gene (word-boundary, case-insensitive). Each sentence is flagged for whether it
    carries a mechanism-of-action cue (inhibit/bind/target/substrate/activate/...)
    versus only association/SNP language, so the verifier and rubric can prefer
    genuine MOA statements. Mechanism-cued sentences are returned first.
    """
    drug = _drug_token(path.get("drug", ""))
    genes = [g for g in path.get("genes", []) if g]
    out: List[Dict] = []
    seen: set = set()
    for a in abstracts:
        text = f"{a.get('title', '')}. {a.get('abstract', '')}"
        for sent in _split_sentences(text):
            low = sent.lower()
            if not (drug and _mentions(drug, low)):
                continue
            hit_genes = [g for g in genes if _mentions(g, low)]
            if not hit_genes:
                continue
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "sentence": sent,
                "source": a.get("source", ""),
                "id": a.get("id", ""),
                "genes": hit_genes,
                "mechanism_cue": bool(_MOA_CUE.search(sent)),
                "association_only": bool(_ASSOC_CUE.search(sent)) and not _MOA_CUE.search(sent),
                "fulltext": bool(a.get("fulltext")),
            })
    # Mechanism-cued sentences first; association-only sentences last.
    out.sort(key=lambda s: (not s["mechanism_cue"], s["association_only"]))
    return out


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


def _llm_prompt(path: Dict, abstracts: List[Dict],
                sentences: Optional[List[Dict]] = None) -> List[Dict]:
    lit = "\n\n".join(
        f"[{a['source']}:{a['id']}] {a['title']}\n{a['abstract'][:1200]}"
        for a in abstracts if a.get("abstract")
    ) or "(no abstracts retrieved)"
    focus = "\n".join(
        f"- [{s['source']}:{s['id']}] {s['sentence']}" for s in (sentences or [])
    ) or "(no sentence co-mentions the drug and a bridge gene)"
    sys = (
        "You are a strict biomedical evidence reviewer. You are given a proposed "
        "drug mechanism-of-action (MOA) path and supporting text. Decide whether the "
        "text SUPPORTS the mechanism. Use ONLY the provided text; do not use outside "
        "knowledge or invent citations.\n\n"
        "GRADING RUBRIC (apply strictly):\n"
        "* 'supported' REQUIRES an EXPLICIT mechanism-of-action statement: the drug "
        "inhibits / activates / binds / antagonizes / is a substrate of / targets the "
        "named protein/gene, OR the named gene/protein IS the drug's molecular target. "
        "The statement must directly connect the drug to the named protein/gene.\n"
        "* A PHARMACOGENETIC or STATISTICAL ASSOCIATION ALONE is NOT mechanism "
        "support and MUST be graded 'weak' (never 'supported'). This includes: a SNP "
        "or sequence variant, a genotype, an 'rsNNNN' identifier, a polymorphism, a "
        "GWAS hit, a drug-resistance association, or a prognosis/survival/risk "
        "association. Such findings describe a correlation, not how the drug acts on "
        "the target.\n"
        "* 'weak' = the drug and gene co-occur but no explicit MOA statement (e.g. "
        "only association, expression, or contextual co-mention).\n"
        "* 'contradicted' = the text states the drug does NOT act on the target.\n"
        "* 'unknown' = the text does not address the drug-target relationship.\n\n"
        "The evidence_quote MUST be a verbatim span copied from the provided text "
        "(prefer the focused mechanism sentences). Respond ONLY as JSON with keys: "
        "grade (one of supported|weak|contradicted|unknown), evidence_quote (<=40-word "
        "verbatim quote from the provided text, or empty), rationale (<=30 words)."
    )
    usr = (
        f"Mechanism path:\n{path['text']}\n\n"
        f"Focused mechanism sentences (drug + bridge gene co-mentions; primary "
        f"evidence):\n{focus}\n\n"
        f"Full abstracts (context only):\n{lit}"
    )
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


def llm_grade(path: Dict, abstracts: List[Dict],
              sentences: Optional[List[Dict]] = None) -> Optional[Dict]:
    if not llm_available():
        return None
    res = _parse_json(
        chat(_llm_prompt(path, abstracts, sentences), json_mode=True, temperature=0.0)
    )
    if not res or res.get("grade") not in GRADES:
        return None
    return {"grade": res["grade"], "evidence": res.get("evidence_quote", ""),
            "rationale": res.get("rationale", "")}


def verify_mechanism(path: Dict, n_lit: int = 5, use_llm: bool = True) -> Dict:
    """Retrieve abstracts for the path and grade mechanistic support.

    Retrieval merges several complementary Europe PMC queries (exact-phrase
    drug+gene, a mechanism-cued variant, optional second bridge gene, and an
    indication query) so the canonical MOA abstract is surfaced more reliably
    than the previous single ``drug AND gene`` query.
    """
    drug = _drug_token(path.get("drug", ""))
    genes = path.get("genes", [])
    disease = path.get("disease")
    abstracts = retrieve_for_mechanism(drug, genes, disease=disease, n=n_lit)

    # Bonus: harvest gene-mentioning passages from OA full text of the top record(s)
    # so an MOA sentence that lives in the body (not the abstract) can still ground
    # the grade. Bounded + graceful: most papers are not OA and yield nothing.
    ft_records = fulltext_passages(abstracts, genes, max_papers=2)
    fulltext_used = bool(ft_records)
    evidence_pool = abstracts + ft_records

    # Sentence-level grounding: isolate drug+gene co-mention sentences as the
    # primary evidence handed to the LLM (abstracts stay as context).
    sentences = mechanism_sentences(path, evidence_pool)

    query = (f'{drug} AND {genes[0]}' if genes else drug) + " [+mechanism/indication variants]"
    lex = lexical_grade(path, abstracts)
    out = {
        "path": path["text"], "type": path.get("type"),
        "query": query, "n_abstracts": len([a for a in abstracts if a.get("abstract")]),
        "lexical": lex, "llm": None,
        "grade": lex["grade"], "source": "lexical",
        "citations": [f"{a['source']}:{a['id']}" for a in abstracts if a.get("id")][:n_lit],
        "evidence_sentences": sentences,
        "fulltext_used": fulltext_used,
    }
    if use_llm:
        llm = llm_grade(path, abstracts, sentences)
        if llm is not None:
            out["llm"] = llm
            out["grade"] = llm["grade"]
            out["source"] = "llm"
    return out
