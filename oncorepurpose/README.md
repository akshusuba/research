# OncoRepurpose-GNN

Honest, deliverable-first **graph neural network drug repurposing for cancer** on
the PrimeKG biomedical knowledge graph.

The project predicts `drug --indication--> cancer` links, rigorously establishes
**when graph topology beats a tuned tabular baseline**, and ships an **agentic LLM
evidence-report** layer that turns top predictions into citation-grounded,
LLM-judged candidate dossiers.

## Why this design (the honest core)

A graph model is only worth its complexity when the prediction genuinely depends
on topology that node features cannot encode. We test that head-on:

- **Shared features.** The same SentenceTransformer node features feed *both* the
  GNN and the XGBoost baseline, so any gap is attributable to graph structure.
- **Leakage-safe splits.** Transductive (random edges) vs **inductive cold-disease**
  (whole cancer diseases held out, TxGNN-style) vs **inductive cold-drug**.
- **Memorization control.** A DistMult KGE has no embedding for unseen nodes and is
  expected to collapse in the inductive regimes.
- **Topology ablation.** Shuffling / removing the message-passing graph must erase
  the GNN's advantage if the win is genuinely topological.

If a tuned XGBoost matched the GNN, that would be reported as the finding. On
PrimeKG repurposing the GNN has a real mechanism to win: it can traverse multi-hop
`drug -> protein -> ... -> disease` paths that a tabular model cannot.

## Results (canonical: 5 seeds, SentenceTransformer features; full data in `results/oncorepurpose.json`)

Test AUROC (mean +/- std over 5 seeds):

| Regime | GNN (ours) | XGBoost | MLP | KGE |
|---|---|---|---|---|
| Transductive | 0.991+/-0.001 | 0.989+/-0.002 | 0.973+/-0.002 | 0.907+/-0.008 |
| Inductive (cold-disease, oncology) | 0.971+/-0.004 | 0.965+/-0.011 | 0.927+/-0.023 | 0.452+/-0.032 |
| Inductive (cold-drug) | 0.964+/-0.005 | 0.958+/-0.006 | 0.930+/-0.011 | 0.482+/-0.031 |

Topology ablation (GNN, cold-disease AUROC): intact 0.974 -> shuffle 0.936 ->
empty 0.773. Relation ablation: removing drug-protein edges hurts most
(0.974 -> 0.966).

Honest reading of the canonical run:
- The **GNN is the best model in every regime**, significantly above MLP and KGE.
- The **DistMult KGE memorizer collapses to ~chance (0.45-0.48) on unseen nodes**,
  while the feature GNN stays ~0.97 -- the clearest demonstration that the task
  needs content + structure, not memorized node identity.
- The **topology ablation** (empty graph 0.773 vs intact 0.974) shows the GNN's
  performance is genuinely driven by graph structure.
- **XGBoost on the same features is a strong baseline** (as the literature warns):
  the GNN beats it significantly in transductive (p=0.018) and cold-drug (p=0.011)
  but only narrowly and non-significantly in cold-disease (p=0.15). We report this
  transparently rather than overclaiming -- the graph's marginal value over a
  well-tuned tabular model is real but modest, while its value over a memorization
  baseline is decisive.

## Layout

```
oncorepurpose/
  config.py              paths, PrimeKG schema, oncology keywords
  datasets.py            load PrimeKG HeteroData + features + target edges
  features.py            shared SentenceTransformer node features (+ hashing fallback)
  models.py              HeteroGNN, FeatureMLP, DistMultKGE, EdgeMLPDecoder
  data/                  download.py, build_graph.py (kg.csv -> HeteroData)
  baselines/             xgboost_baseline.py (tuned tabular control)
  evaluation/            splits.py, metrics.py, statistical_tests.py, trainer.py
  interpret/             paths.py (candidate ranking + multi-hop KG rationales)
  agent/                 llm.py, evidence_report.py (RAG + LLM-as-judge)
scripts/
  run_experiment.py      canonical 4-model x 3-regime + ablations
  generate_report.py     deliverable: vetted oncology repurposing shortlist
```

## Setup & run

```bash
pip install -r requirements.txt
python -m oncorepurpose.data.download          # PrimeKG kg.csv (~980 MB)
python -m oncorepurpose.data.build_graph       # -> data/primekg_hetero.pt
python scripts/run_experiment.py               # 5 seeds + ablations
ONCO_LLM_API_KEY=sk-... python scripts/generate_report.py \
    --diseases glioblastoma "pancreatic cancer" --top-k 5
```

Set `ONCO_LLM_API_KEY` (and optionally `ONCO_LLM_BASE_URL`, `ONCO_LLM_MODEL`) to
enable the LLM evidence dossiers + LLM-as-judge; without it the report still
includes model scores, KG rationales, and retrieved literature.

## Positioning vs prior work

See [docs/POSITIONING.md](docs/POSITIONING.md). In short: TxGNN (Zitnik, Nature
Medicine 2024) is the KG-repurposing incumbent and is treated as a reference. Our
contribution is the **rigorous "is the graph necessary?" analysis** (tuned XGBoost
baseline + topology ablation), the **agentic evidence-report deliverable**, and a
**focused oncology scope** -- not a new architecture.

*All predictions are hypothesis-generating and not medical advice.*
