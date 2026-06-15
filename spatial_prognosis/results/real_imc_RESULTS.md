# Real-data validation: Jackson-Fischer 2020 breast-cancer IMC (Basel cohort)

**Question:** does the spatial *arrangement* of the tumor microenvironment predict
outcome *beyond* cell composition? Spatial GNN (SAGE) vs composition-only baselines
(LogReg / XGBoost / MLP on cell-type proportions).

## Data
- Source: Zenodo `10.5281/zenodo.3518284` (mirror record `4607374`).
  - `singlecell_locations.zip` -> `Basel_SC_locations.csv` (per-cell X/Y + core id)
  - `singlecell_cluster_labels.zip` -> `Basel_metaclusters.csv` + `Metacluster_annotations.csv`
    (26 marker-derived cell-type metaclusters)
  - `Basel_PatientMetadata.csv` (grade + overall survival) extracted from the 36 GB
    `SingleCell_and_Metadata.zip` via HTTP range requests (`remotezip`) - no full download.
- Cohort: 376 Basel cores, 844,498 cells, 26 cell types. Subsampled to <=1500 cells/core.
- Node features = one-hot(cell type); edges = spatial kNN (k=6). Baselines see cell-type
  proportions (composition). Markers (4.9 GB long-format `SC_dat.csv`) not downloaded;
  metaclusters already encode marker-derived phenotype.
- Trainer/metrics are binary, so both labels are binary: grade G3 vs G1/G2 (372 graphs,
  166 vs 210); 5-yr survival via `binarize_survival` (310 graphs, 249 survived vs 65 died).

## Results (test, mean over 3 seeds)

### Grade (G3 vs G1/G2)
| model   | accuracy | macro-F1 | AUROC |
|---------|----------|----------|-------|
| LogReg  | 0.613    | 0.588    | 0.687 |
| XGBoost | 0.613    | 0.609    | 0.632 |
| MLP     | 0.640    | **0.633**| 0.671 |
| **SAGE GNN** | 0.608 | 0.571 | 0.673 |

Ablation (GNN macro-F1): intact 0.571 / shuffled 0.543 / **empty 0.573** / xgb 0.609.

### 5-year survival
| model   | accuracy | macro-F1 | AUROC |
|---------|----------|----------|-------|
| LogReg  | 0.801    | 0.445    | 0.635 |
| XGBoost | 0.790    | 0.503    | 0.610 |
| MLP     | 0.801    | 0.445    | 0.563 |
| **SAGE GNN** | 0.806 | **0.560** | **0.655** |

Ablation (GNN macro-F1): intact 0.555 / shuffled 0.545 / empty 0.445 / xgb 0.503.

### Synthetic reference (for contrast)
GNN intact macro-F1 1.000; shuffled 0.325; empty 0.325; baselines 0.34-0.51.

## Verdict: NO added value from spatial arrangement on real data
- **Grade:** GNN does *not* beat composition baselines (MLP 0.633 > GNN 0.571). The
  no-edge "empty" GNN (0.573) matches the intact GNN -> edges add nothing. Null.
- **Survival:** GNN has a *modest* edge (macro-F1 0.560, AUROC 0.655), but the
  graph-shuffle ablation does **not** collapse it: shuffled edges (0.545) ~= intact
  (0.555), while only removing edges entirely (empty 0.445) drops performance. So the
  small lift is non-spatial neighborhood feature-aggregation/regularization, *not*
  arrangement. The falsification test fails to attribute the signal to arrangement.
- This is the opposite of the synthetic cohort, where shuffling collapses the GNN
  (1.000 -> 0.325). On this real cohort, arrangement carries no detectable signal
  beyond composition with this setup.

## Caveats
- Trainer is binary-only (`predict_proba[:,1] >= 0.5`); 3-class grade was dichotomized
  to keep baselines fair (3-class would cripple them and falsely inflate the GNN).
- Survival is class-imbalanced (79% survived); most models default to majority class.
- Splits are per-core; 285 patients have 376 cores, so a few patients span folds
  (minor leakage). Cells capped at 1500/core. Markers excluded (cell-type one-hot only).
- Basel cohort only; Zurich not run.
