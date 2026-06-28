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
paths, while random drug–cancer pairs mostly yield no mechanistic path**. (Spurious
links through promiscuous carrier hubs like albumin are now filtered out by the
extractor — see *Hub down-weighting* below.) So the graph's value is **mechanism
and explanation**, not the link score — and that is the GNN's real job here.

Reproduce: `PYTHONPATH=. python scripts/mechanism_demo.py`.

## Finding 3 — mechanism structure separates true indications from random pairs

The falsifiable claim (Aim 4), tested LLM-free on the graph mechanism signal over
400 true oncology indications vs 400 random drug–cancer pairs
(`scripts/evaluate_mechanism.py`, `results/mechanism_eval.json`):

| Metric | True indications | Random pairs |
|---|---|---|
| Mean mechanism score | **1.95** | 0.17 |
| Direct-target rate | **34.0%** | 0.25% |
| Any mechanistic path | **81.3%** | 8.8% |

**Separation AUROC (true vs random): 0.879.** True indications are ~136× more
likely to have a direct drug-target → cancer-gene link, and a random pair almost
never does (figure: `figures/mechanism_eval.png`). So the mechanism signal the
graph provides is real and discriminative — the thing XGBoost's link score cannot
give you.

**Evidence verification (Aim 3).** The verifier (`agent/verify.py`) retrieves
Europe PMC *abstracts* (not just titles) and grades each path
**supported / weak / contradicted / unknown** — via an LLM (OpenRouter
`gpt-4o-mini`) when `ONCO_LLM_API_KEY` is set, and a lexical-grounding fallback
otherwise. Run over 50 true vs 50 random pairs
(`scripts/verify_llm_eval.py`, `results/verify_llm_eval.json`):

- **Random pairs:** 47/50 have *no mechanistic path*; the LLM marks only 1/50 *supported*.
- **True pairs:** LLM marks **11** *supported*, **29** *weak*, 1 *unknown*, 9 *no-path*.
  A stricter rubric (an explicit drug→target MOA statement is required; a SNP/genotype/
  prognosis association is graded *weak*, not *supported*) plus sentence-level grounding
  (only drug+gene co-mention sentences are fed to the model) collapsed *unknown* from 14→1
  and pushed tangential evidence to *weak*.
- **The LLM "supported" calls are now far more precise than the lexical fallback.**
  Of the *supported* true pairs DrugMechDB covers, **LLM = 0.857 (6/7)** have the path
  gene in the curated MOA set, vs **lexical = 0.591 (13/22)**. The LLM is the conservative,
  trustworthy reviewer; lexical co-mention over-calls *supported*. Examples are mechanistic:
  Topotecan→TOP1 (*"a topoisomerase I (TOP1) inhibitor"*), Brigatinib→ALK (*"an ALK tyrosine
  kinase inhibitor"*), Cisplatin→ATP7B (*"downregulates ATP7B … efflux of … cisplatin"*).
- **Remaining limitation:** open-access full text was fetched where available but the
  PMC OA subset covered ~none of these papers, so abstract-level grounding still dominates.

**Curated-mechanism agreement (DrugMechDB).** Mapping DrugMechDB's UniProt
accessions to HGNC symbols (via mygene.info, cached in `data/uniprot2symbol.json`)
makes the comparison meaningful. On the 263 true pairs whose drug DrugMechDB
covers, our extracted bridge genes overlap the curated MOA genes for **211 pairs
(agreement 0.802)** — e.g. Methotrexate→{ATIC, DHFR, TYMS}, Topotecan→{TOP1},
Decitabine→{DNMT1, DNMT3A}. So the paths we extract align with an independent,
expert-curated mechanism resource.

**Hub down-weighting (Aim 2 hardening).** Pure plasma carriers (albumin, A2M, …)
are hard-excluded from paths, and every bridge is softly down-weighted by an
IDF-style promiscuity penalty (`1/log(drug-degree)`). This removes spurious carrier
bridges (e.g. `Sulfamerazine→ALB→…→AML`) while *improving* both headline metrics
(separation AUROC 0.878→0.879, DrugMechDB agreement 0.787→0.802).

## Finding 4 — the graph's one genuine edge: blinded mechanism recovery

We trained the GNN jointly (link BCE + a contrastive loss that ranks the curated
DrugMechDB bridge gene above degree-matched decoys) and asked it to *name the
bridge gene* for held-out cold-disease pairs (3 seeds, n≈186–260;
`scripts/evaluate_mechanism_recovery.py`, `results/mechanism_recovery_eval.json`).
This is an axis a tabular model cannot touch — XGBoost never embeds a third (gene)
node. Read it honestly:

| Bridge-gene recovery | Unblinded R@10 | Blinded R@10 | Blinded MRR |
|---|---|---|---|
| Joint GNN (link + mechanism) | 0.31 | **0.25** | **0.21** |
| Trivial target-lookup | **0.80** | 0.00 | 0.00 |
| Link-only GNN (same head, no aux loss) | 0.00 | 0.00 | 0.00 |
| Degree prior | 0.02 | 0.02 | 0.02 |

**Unblinded**, the curated bridge gene is usually just the drug's direct target, so
a trivial "look up the drug's own targets" baseline dominates (R@10 0.80) and the
GNN adds nothing. **But with the direct drug→target edge removed (mechanism-blinded),
the trivial lookup and degree prior collapse to ~0, while the joint GNN still
recovers the bridge gene** (R@10 0.25 ± 0.02, per-seed 0.23–0.26) from indirect
structure. A link-only GNN scored through the *identical* mechanism head (no
auxiliary loss) also recovers nothing, so the signal is attributable to the
auxiliary objective, not the architecture or scorer. Caveat: supervision is
drug-level and the gene may stay reachable via retained PPI/pathway edges, so this
is recovery of *known* mechanism via indirect paths, not de novo discovery — a real
but narrow graph-only advantage, on a modest sample.

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
    uniprot_map.py       UniProt accession -> HGNC symbol (mygene.info, cached)
  agent/
    llm.py               provider-agnostic chat client (cached)
    evidence_report.py   Europe PMC retrieval (abstracts) + LLM-as-judge dossier
    retrieval.py         multi-query Europe PMC MOA retrieval (merge + rank)
    fulltext.py          open-access full-text fetch (Europe PMC OA subset)
    verify.py            verifier: strict MOA rubric, sentence grounding, LLM + lexical
scripts/
  run_experiment.py      4-model x 3-regime comparison + ablations
  mechanism_demo.py      multi-hop mechanism paths for oncology pairs
  evaluate_mechanism.py  true-vs-random separation + grounding + DrugMechDB (Aim 4)
  generate_report.py     mechanism-grounded oncology repurposing shortlist (hypothesis-generating)
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
- [x] Verifier reads Europe PMC **abstracts** (not titles); LLM grade + lexical fallback.
- [x] Quantitative true-vs-random separation (AUROC 0.879); the falsifiable claim holds.
- [x] Hard-negative stress test: strong vs random / oncology-drug negatives (0.887 / 0.870), **modest (0.609) against shared-target negatives** -- honest about where the mechanism signal is and isn't decisive.
- [x] LLM verifier run (OpenRouter `gpt-4o-mini`); stricter than lexical, separates true vs random.
- [x] Improved multi-query MOA retrieval: gene-mention 80%→93%; LLM *supported* 8→14 / 50.
- [x] DrugMechDB agreement via UniProt→HGNC map (mygene.info): **0.802** on covered pairs.
- [x] Stricter MOA rubric + sentence grounding: LLM-*supported* precision **0.857** vs lexical 0.591.
- [x] Hub down-weighting (carrier exclusion + IDF promiscuity penalty): removes albumin bridges, AUROC 0.878→0.879.
- [x] Open-access full-text fetch wired in (PMC OA subset is sparse, so abstracts still dominate).
- [x] Joint mechanism supervision creates recoverable mechanism signal: the jointly-trained graph names the held-out bridge gene at R@10 0.25, while degree-trivial and link-only baselines (no mechanism objective) hit 0 -- the signal comes from the mechanism objective, not architecture alone.
- [x] Mechanism-first shortlist generator: ranks by specificity lift, keeps only candidates with a real MOA path (no phenotype bridges); see `results/repurposing_shortlist.md`.
- [x] Learning-track notebooks: Parts 7--8 are executed with saved outputs (real numbers); Parts 1--6 and the full self-contained notebook are runnable templates.
- [ ] Broaden full-text coverage (non-OA sources) and scale the LLM verifier run.

*All predictions are hypothesis-generating and not medical advice.*
