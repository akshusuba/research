# SpatialGNN — When the Neighborhood *Is* the Label

> A graph neural network only earns its complexity when the prediction target
> depends on structure that a node's own features cannot encode. **Spatial
> tissue-domain identification is exactly that task**: a cell's domain is a
> property of *where it sits*, not just what it expresses — so a spatial GNN
> structurally beats strong non-graph models (MLP, XGBoost), and this repo
> shows it cleanly.

## Why this project exists

It is the result of a deliberate, evidence-driven search. We first tried
synthetic-lethality link prediction; with strong baselines and real features, a
GNN merely *tied* XGBoost/MLP — the graph wasn't pulling its weight (see
`../synlethgnn`). The lesson: pick a task where **the label is defined by the
neighborhood**. Spatial-omics domain mapping is that task, and the GNN advantage
is dramatic and provable.

## Headline result (synthetic tissue benchmark, multi-seed)

Cross-section evaluation (whole tissues held out — absolute position cannot be
memorized, so only neighborhood-relative reasoning generalizes):

| Cross-section (5 seeds) | XGBoost | MLP | **Spatial GNN (ours)** |
|---|---|---|---|
| Macro-F1 | 0.859 | 0.889 | **0.987** |
| ARI | 0.702 | 0.756 | **0.974** |

**Graph-removal ablation** (the falsification test): the identical GNN on a
*shuffled* graph (F1 0.886) or *empty* graph (0.887) collapses to the MLP
(0.889), while on the intact spatial graph it reaches **0.987**. The advantage
is **spatial**, not features — and the baselines are strong (~0.86–0.89 F1, far
from chance), so this is a fair, decisive win.

(Exact multi-seed numbers are in `results/`; figures in `figures/`.)

## The idea

In spatial transcriptomics (10x Visium, MERFISH, Xenium), each cell/spot has an
expression vector *and* a 2D location. Tissue **domains** (e.g., cortical
layers, tumor niches) are spatially contiguous and shared among neighbors, while
per-cell expression is noisy and dropout-corrupted. So:

- An **MLP / XGBoost** sees one cell's noisy expression → bounded accuracy.
- A **spatial GNN** aggregates the neighborhood → denoises and recovers the
  domain → wins, especially across tissue sections where position can't be
  memorized.

Coordinates are **never** given as features — spatial information enters *only*
through the graph, so any GNN gain is attributable to message passing.

## Install & run

```bash
cd research/spatialgnn
pip install -r requirements.txt        # or reuse ../celiac/venv

python scripts/run_synthetic_demo.py          # full: 5 seeds (offline)
python scripts/run_synthetic_demo.py --fast    # quick smoke test
python scripts/make_figures.py                 # figures from results/
```

Real data (any AnnData `.h5ad` with spatial coords + a domain-label column):

```bash
python scripts/run_real.py --h5ad dlpfc.h5ad \
    --label-key layer_guess_reordered --sample-key sample_id
```

The recommended benchmark is the **LIBD DLPFC** Visium dataset (12 sections,
manual cortical-layer annotations — the field's ground truth for spatial-domain
methods). It ships via R/Bioconductor; export it once to `.h5ad`:

```r
library(spatialLIBD); library(zellkonverter)
spe <- fetch_data("spe")            # 12 DLPFC sections with layer labels
writeH5AD(spe, "dlpfc.h5ad")
```

Any 10x Visium / MERFISH / Xenium AnnData works the same way.

## Project layout

```
spatialgnn/
├── README.md
├── requirements.txt
├── docs/
│   └── proposal.tex             # SRI proposal (colored figures, tables, formulas)
├── spatialgnn/
│   ├── config.py                # dataclass configs
│   ├── data/
│   │   ├── synthetic.py         # neighborhood-defined domain benchmark
│   │   └── real.py              # AnnData .h5ad spatial loader (DLPFC/MERFISH/Xenium)
│   ├── models/
│   │   ├── gnn.py               # spatial GraphSAGE/GAT/GCN node classifier (ours)
│   │   ├── mlp.py               # structure-blind MLP control
│   │   └── xgboost_baseline.py  # strong tabular baseline
│   ├── splits.py                # cross-section + within-section (leakage-safe) splits
│   ├── metrics.py               # accuracy, macro-F1, ARI
│   ├── train.py                 # shared node-classification trainer
│   ├── experiment.py            # multi-seed comparison
│   └── ablation.py              # graph-removal falsification test
├── scripts/
│   ├── run_synthetic_demo.py
│   ├── run_real.py
│   └── make_figures.py
├── results/                     # JSON outputs + run log
└── figures/                     # generated figures
```

## Why the comparison is fair

Every model (GNN, MLP) shares the same trainer (optimizer, loss, early stopping,
metrics); XGBoost gets the same per-cell features. The MLP and XGBoost use the
*same* expression features the GNN starts from. The only difference is whether
the model can see the spatial graph — so the gap is the value of spatial message
passing, nothing else.

## Translational relevance

Spatial-domain maps are foundational for understanding tissue organization in
disease — cortical-layer disruption in neuropsychiatric/neurodegenerative
conditions, and tumor-microenvironment niches in cancer. Accurate, label-
efficient domain mapping helps localize where disease-associated changes occur.
Predictions are research-only and hypothesis-generating.
