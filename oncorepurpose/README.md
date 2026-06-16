# OncoEvidence — Mechanism-Guided AI Evidence Triage for Cancer Drug Repurposing

> **Core question.** For cancer drug repurposing, can a biomedical knowledge graph
> plus a citation-grounded LLM agent prioritize candidates by *both* predicted
> therapeutic relevance *and* mechanistic plausibility — and when does the graph
> add value over a strong tabular baseline?

OncoEvidence is an oncology-focused **evidence-triage pipeline** built on the
PrimeKG biomedical knowledge graph. It ranks drug–cancer repurposing candidates,
extracts the multi-hop **mechanism-of-action paths** that justify each one, and
uses an LLM to grade whether those paths are supported by the literature. The
project is deliberately honest about where graph learning helps and where a tuned
tabular model is already enough.

## Specific aims

1. **Candidate generation.** Rank `drug --indication--> cancer` pairs with
   PrimeKG models (heterogeneous GNN, DistMult KGE) against a **tuned XGBoost**
   on the *same* node features, and report honestly where the graph helps.
2. **Mechanism extraction.** For top candidates, extract multi-hop
   `drug → target protein → (PPI / pathway) → cancer gene → cancer` paths,
   prioritizing genuine mechanism relations over phenotype/symptom coincidence.
3. **LLM evidence verification.** Retrieve literature and have an LLM grade each
   proposed path as **supported / weak / contradicted / unknown**, with quoted
   evidence (an evidence reviewer, not a hypothesis generator).
4. **Evaluation.** Test against known indications and curated mechanism resources
   (e.g. DrugMechDB, where covered). Falsifiable claim: *LLM-verified mechanism
   paths separate true indications from random drug–cancer pairs better than the
   link score alone, and the verifier keeps real MOA paths while rejecting
   coincidental phenotype/hub bridges.*

## Finding 1 — the link task alone does **not** need the graph

With a properly **tuned** XGBoost on shared SentenceTransformer features, the GNN
no longer wins anywhere (corrected run, 5 seeds; `results/oncorepurpose.json`).
The earlier "GNN wins" was an artifact of an untuned baseline plus a
message-passing leak, both fixed.

| Regime | GNN | tuned XGBoost | MLP | KGE |
|---|---|---|---|---|
| Transductive | 0.977 | **0.988** | 0.973 | 0.907 |
| Inductive (cold-disease, oncology) | 0.882 | **0.963** | 0.930 | 0.452 |
| Inductive (cold-drug) | 0.956 | **0.958** | 0.931 | 0.482 |

Test AUROC. The honest reading: for *link ranking* with rich text features, a
strong tabular model is as good or better. A drug/disease name embedding lets
XGBoost take a "semantic similarity" shortcut without any graph reasoning. This
is what motivates the mechanism-aware reframing below.

## Finding 2 — but the graph carries real mechanism a tabular model cannot

XGBoost can score a pair; it **cannot** produce a traceable mechanism. The graph
can. On true oncology indications, the multi-hop extractor recovers textbook
direct-target mechanisms (`oncorepurpose/interpret/mechanism_paths.py`):

```
Quizartinib  --targets--> FLT3  <--associated-- myeloid leukemia          (FLT3 inhibitor)
Tamibarotene --targets--> RARA  <--associated-- acute promyelocytic leukemia (RARA driver)
Etoposide    --targets--> TOP2A --interacts--> RB1 <--associated-- small cell lung carcinoma
Trifluridine --targets--> TYMS  <--associated-- colorectal cancer         (thymidylate synthase)
```

In a quick check, **true indications reliably yield specific direct-target
paths, while random drug–cancer pairs mostly yield no mechanistic path** (the few
that do go through promiscuous hubs like albumin — exactly the coincidental links
the LLM verifier is meant to reject). So the graph's value is **mechanism and
explanation**, not the link score — and that is the GNN's real job here.

Reproduce: `PYTHONPATH=. python scripts/mechanism_demo.py`.

## Positioning (honest novelty)

This project does **not** claim to invent KG-based or LLM-based drug repurposing.
Close prior work exists and is treated as reference:

- **TxGNN** (Zitnik lab, *Nature Medicine* 2024) — KG GNN for zero-shot
  repurposing with multi-hop explanations across 17,080 diseases.
- **KGML-xDTD** — drug-treatment prediction plus KG path-based mechanism
  descriptions.
- **DrugKLM** — biomedical KGs combined with LLM mechanistic reasoning for
  therapeutic prioritization.
- **Decagon** (Stanford) — GNNs for relational drug-pair (polypharmacy) tasks,
  where the graph clearly beats tabular models.

> Prior work has used biomedical knowledge graphs and GNNs for drug repurposing,
> and newer work is beginning to fuse KGs with LLM reasoning. OncoEvidence builds
> and **evaluates** an oncology-specific, citation-grounded evidence-triage
> pipeline that (a) tests honestly when graph structure is actually necessary, and
> (b) checks whether proposed mechanism paths are supported by retrieved
> literature. We *combine and evaluate*; we do not claim to be first.

See [docs/POSITIONING.md](docs/POSITIONING.md) for detail.

## Layout

```
oncorepurpose/
  config.py              paths, PrimeKG schema, oncology keywords
  datasets.py            load PrimeKG HeteroData + features + target edges
  features.py            shared SentenceTransformer node features (+ hashing fallback)
  models.py              HeteroGNN, FeatureMLP, DistMultKGE, EdgeMLPDecoder
  data/                  download.py, build_graph.py (kg.csv -> HeteroData)
  baselines/             xgboost_baseline.py (Optuna-tuned tabular control)
  evaluation/            splits.py, metrics.py, statistical_tests.py, trainer.py
  interpret/
    paths.py             2-hop bridge rationales + candidate ranking
    mechanism_paths.py   multi-hop MOA path extractor (direct-target / PPI / pathway)
  agent/                 llm.py, evidence_report.py (literature + LLM-as-judge)
scripts/
  run_experiment.py      4-model x 3-regime comparison + ablations
  mechanism_demo.py      multi-hop mechanism paths for oncology pairs
  generate_report.py     deliverable: vetted oncology repurposing shortlist
```

## Setup & run

```bash
pip install -r requirements.txt
python -m oncorepurpose.data.download          # PrimeKG kg.csv (~980 MB)
python -m oncorepurpose.data.build_graph       # -> data/primekg_hetero.pt
PYTHONPATH=. python scripts/run_experiment.py  # 5 seeds + ablations (corrected)
PYTHONPATH=. python scripts/mechanism_demo.py  # mechanism paths
ONCO_LLM_API_KEY=sk-... PYTHONPATH=. python scripts/generate_report.py \
    --diseases glioblastoma "pancreatic cancer" --top-k 5
```

Set `ONCO_LLM_API_KEY` (and optionally `ONCO_LLM_BASE_URL`, `ONCO_LLM_MODEL`) to
enable the LLM evidence dossiers + LLM-as-judge; without it the report still
includes model scores, mechanism paths, and retrieved literature.

## Status / roadmap

- [x] Corrected, leakage-safe benchmark (tuned XGBoost); honest negative result on the link task.
- [x] Multi-hop mechanism-path extractor; validated on real oncology indications.
- [ ] Upgrade LLM verifier to read **abstracts/passages** (true citation grounding), not titles.
- [ ] Stronger hub down-weighting (e.g. albumin-type promiscuous bridges).
- [ ] Quantitative evaluation vs DrugMechDB + true/random separation (the falsifiable claim).

*All predictions are hypothesis-generating and not medical advice.*
