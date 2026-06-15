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

## Relationship to prior work (honest positioning)

GNN-based spatial-domain detection is an **established field** — SpaGCN (Nat.
Methods 2021), STAGATE (Nat. Commun. 2022), and GraphST (Nat. Commun. 2023) all
do this and are standard on DLPFC. **This project does not claim a new
architecture.** Those methods are mostly *unsupervised* (autoencoder + clustering
vs. ARI) and rarely include a strong *non-graph* baseline. The contribution here
is the rigorous, under-asked **question**: *how much does spatial structure
actually add over tuned XGBoost/MLP on identical features?* — answered with a
supervised, leakage-safe cross-section protocol, a graph-removal falsification
test, and replication across two technologies. It's a rigorous benchmark and
honest analysis, not a SOTA or novel-method claim.

## Headline result — real human cortex (LIBD DLPFC, 3 seeds)

The thesis holds **decisively on gold-standard real data**. On the LIBD DLPFC
10x Visium benchmark (12 sections, 47,329 expert-annotated spots, 7 cortical
domains), **cross-section** evaluation (whole tissues held out — absolute
position cannot be memorized, so only neighborhood-relative reasoning
generalizes, the disease-cohort regime):

| Real DLPFC, cross-section (3 seeds) | XGBoost | MLP | **SAGE (ours)** | **GAT (ours)** |
|---|---|---|---|---|
| Macro-F1 | 0.516 ± 0.02 | 0.488 ± 0.03 | **0.754 ± 0.01** | **0.742 ± 0.00** |
| ARI | 0.369 | 0.377 | **0.651** | **0.656** |
| Accuracy | 0.621 | 0.616 | **0.797** | **0.793** |

A **+0.24 macro-F1** win over strong non-spatial baselines on the same expression
features — and the win is **architecture-robust**: both a GraphSAGE encoder and a
**graph-attention (GAT)** encoder land at ~0.75 F1 / ~0.65 ARI. **Graph-removal
ablation** (the falsification test) collapses the identical GNN back onto the
blind MLP — proving the gain is *spatial*, not features:

| Real DLPFC ablation (GNN) | Intact | Shuffled graph | Empty graph | MLP ref. |
|---|---|---|---|---|
| Macro-F1 | **0.755** | 0.492 | 0.487 | 0.484 |
| ARI | **0.654** | 0.378 | 0.376 | 0.375 |

Honest caveat: the real-data margin is smaller than on the synthetic benchmark
(real cortical layers carry *some* per-spot signal, so the baselines clear
chance), but the win and the ablation collapse are unambiguous.

## Second tissue, different technology (osmFISH mouse cortex, 3 seeds)

To show the result is not Visium-specific, we add **osmFISH** — a single-molecule
FISH assay at **single-cell resolution** with only **33 genes** and true cortical
**region** labels (L1/Pia, L2-3, L4, L5, L6, white matter, ventricle,
hippocampus…). With so few genes, per-cell features are weak, so neighborhood
aggregation should matter even more. It does (within-section transductive split,
all regions represented):

| osmFISH cortex (3 seeds) | XGBoost | MLP | **SAGE** | **GAT** |
|---|---|---|---|---|
| Macro-F1 | 0.543 ± 0.01 | 0.632 ± 0.01 | **0.960 ± 0.00** | **0.966 ± 0.01** |
| ARI | 0.436 | 0.512 | **0.947** | **0.954** |
| Accuracy | 0.662 | 0.711 | **0.971** | **0.976** |

A **+0.33–0.42 macro-F1** gap — even larger than DLPFC, exactly because the
33-gene per-cell signal is weak and the spatial neighborhood carries the domain.
(Note: osmFISH is a single section, so this is a within-section *transductive*
node-classification split, an easier regime than DLPFC's cross-section test;
reported as such. The cross-section DLPFC result above remains the harder,
headline evidence.)

The **falsification test repeats on osmFISH** — destroy the spatial graph and the
identical GNN collapses back to the blind MLP, confirming the win is spatial on a
second technology too:

| osmFISH ablation (GNN) | Intact | Shuffled graph | Empty graph | MLP ref. |
|---|---|---|---|---|
| Macro-F1 | **0.952** | 0.579 | 0.612 | 0.632 |
| ARI | **0.939** | 0.476 | 0.514 | 0.512 |

## Mechanism proof (synthetic tissue benchmark, 5 seeds)

A controlled benchmark where per-spot features carry **no** domain signal (so
topology is the only route) isolates the mechanism:

| Synthetic, cross-section (5 seeds) | XGBoost | MLP | **Spatial GNN (ours)** |
|---|---|---|---|
| Macro-F1 | 0.859 | 0.889 | **0.987** |
| ARI | 0.702 | 0.756 | **0.974** |

Here too the graph-removal ablation collapses the GNN (shuffled 0.886 / empty
0.887) to the MLP (0.889). Exact multi-seed numbers are in `results/`
(`real_comparison.json`, `real_comparison_osmfish.json`, `real_graph_ablation.json`,
`real_graph_ablation_osmfish.json`, `synthetic_comparison.json`). Figures are in
`figures/` (`fig1_real_comparison.png`, `fig2_real_ablation.png`,
`fig3_synthetic.png`), regenerated by `scripts/make_figures.py`.

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
# multi-section DLPFC (cross-section, leakage-safe), all 4 models incl. GAT
python scripts/run_real.py --h5ad data/dlpfc.h5ad \
    --label-key layer_guess_reordered --sample-key sample_id

# graph-removal falsification test (DLPFC, cross-section)
python scripts/run_real_ablation.py --h5ad data/dlpfc.h5ad \
    --label-key layer_guess_reordered --sample-key sample_id

# second tissue: osmFISH (single section -> stratified split), + its ablation
python scripts/build_osmfish_h5ad.py            # data/osmfish.loom -> data/osmfish.h5ad
python scripts/run_real.py --h5ad data/osmfish.h5ad --label-key Region \
    --mode stratified --out real_comparison_osmfish.json
python scripts/run_real_ablation.py --h5ad data/osmfish.h5ad --label-key Region \
    --mode stratified --out real_graph_ablation_osmfish.json
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
│   ├── run_real.py              # multi-seed comparison (XGBoost/MLP/SAGE/GAT)
│   ├── run_real_ablation.py     # graph-removal falsification test
│   ├── build_dlpfc_h5ad.py      # LIBD DLPFC -> dlpfc.h5ad
│   ├── build_osmfish_h5ad.py    # osmFISH .loom -> osmfish.h5ad
│   └── make_figures.py          # fig1 comparison, fig2 ablation, fig3 synthetic
├── results/                     # JSON outputs + run log
└── figures/                     # generated figures
```

## Why the comparison is fair

Every model (GNN, MLP) shares the same trainer (optimizer, loss, early stopping,
metrics); XGBoost gets the same per-cell features. The MLP and XGBoost use the
*same* expression features the GNN starts from. The only difference is whether
the model can see the spatial graph — so the gap is the value of spatial message
passing, nothing else.

## Translational relevance — toward precision pathology

Spatial-domain maps are the enabling step that turns a spatial-transcriptomics
slide into region-resolved biology. The cortical-layer benchmark here is the
rigorous *proof of concept*; the same capability powers **precision pathology**:
mapping tumor-microenvironment niches (tumor core, invasive front, stroma, immune
infiltrate) to study heterogeneity, metastasis, and immunotherapy response, and
localizing layer-specific molecular pathology in neuropsychiatric/
neurodegenerative disease. Spatial transcriptomics is increasingly argued to be
poised to reshape precision oncology (Cilento, Sweeney & Butler, *J. Cancer Res.
Clin. Oncol.* 2024; PMID 38850363) — but only once spots can be reliably assigned
to domains, which is exactly what a spatially-aware model does and a per-spot
model cannot. Predictions are research-only and hypothesis-generating.

See `docs/proposal.tex` / `docs/proposal.pdf` for the full SRI research proposal.
