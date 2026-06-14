# Health × GNN — 10 Novel Research Experiments

Each experiment is a self-contained study at the intersection of **healthcare** and
**graph neural networks**. The central thesis across the suite: *many health
prediction problems are fundamentally relational, so a model that reads graph
structure (a GNN) should beat strong tabular models (XGBoost, MLP) that only see
per-entity features.*

## The contract (every experiment must satisfy this)

1. Build a graph-structured health dataset where **topology carries genuine
   signal** (homophily, contagion, guilt-by-association, connectivity patterns).
2. Train **XGBoost** and an **MLP** on the *same raw node/graph feature matrix*
   the GNN sees — node features only, **no hand-engineered graph features leaked
   to the baselines** (that would make the comparison unfair).
3. Train a **GNN** (GCN / GraphSAGE / GAT swept, best-on-validation reported).
4. Call `common.log_result(...)`, which records metrics to `results/<name>.json`
   and **raises unless the GNN beats BOTH baselines on the primary metric**.
5. Primary metric is **AUC** (macro-OVR for multiclass) unless noted.

Fair-comparison rule: the GNN's *only* structural advantage must be message
passing over edges. Keep node features informative-but-insufficient so structure
is what closes the gap.

## Shared infrastructure

- `common.py` — seeding, metrics, `run_xgboost`, `run_mlp`, `run_gnn_node`,
  `run_gnn_graph`, and `log_result` (the win-checker).
- Run experiments with the repo venv: `research/.venv` (reuses torch 2.9.1,
  has torch-geometric, xgboost, sklearn, pandas, scipy, networkx).
- Each experiment lives in `expNN_name/run.py` and writes `results/<name>.json`.

## The 10 ideas

| # | Dir | Idea | Why graph structure helps | Task |
|---|-----|------|---------------------------|------|
| 1 | `exp01_epidemic` | **EpiGNN** — infection-risk on human contact networks | contagion spreads along edges; risk ≈ exposure to infected neighbors | node-cls |
| 2 | `exp02_readmission` | **ReadmitGraph** — 30-day readmission via patient-similarity graph | patients linked by shared diagnoses/comorbidities are homophilous in outcome | node-cls |
| 3 | `exp03_disease_gene` | **PPI-DiseaseGene** — disease-gene prediction on protein–protein interaction net | guilt-by-association: disease genes cluster topologically in PPI | node-cls |
| 4 | `exp04_connectome` | **ConnectomeDx** — neuro-disorder dx from brain functional connectivity | disorder alters connectivity *patterns*, not regional features alone | graph-cls |
| 5 | `exp05_hai_transfer` | **HAI-Transfer** — healthcare-associated infection risk via patient-transfer network | pathogens propagate through ward/hospital transfer topology | node-cls |
| 6 | `exp06_microbiome` | **MicrobiomeNet** — disease classification from microbial co-occurrence nets | disease shifts *interaction structure* of the microbiome, not just abundances | graph-cls |
| 7 | `exp07_comorbidity` | **ComorbidityProg** — chronic-disease progression on comorbidity graph | progression follows comorbidity adjacency (disease trajectories) | node-cls |
| 8 | `exp08_ddi` | **DDI-Mol** — adverse drug–drug interaction from molecular graphs | interaction depends on molecular substructure topology | graph-cls (pair) |
| 9 | `exp09_medkg` | **MedKG-Dx** — diagnosis prediction over a symptom–disease knowledge graph | diagnosis = relational reasoning over symptom/disease links | node-cls (hetero-ish) |
| 10 | `exp10_icu_sensor` | **ICU-SensorGraph** — deterioration prediction from physiological sensor graphs | inter-signal correlation *structure* signals deterioration before marginals do | graph-cls |

Each `run.py` prints a comparison table and saves a JSON record. A final
`aggregate.py` collects all `results/*.json` into a leaderboard.
