# spatial_prognosis — Does Tumor *Arrangement* Predict Outcome Beyond *Composition*?

> A clinical question, answered rigorously. The tumor microenvironment's
> prognostic value is widely attributed to *how* immune and tumor cells are
> spatially organized (e.g., immune **infiltration vs. exclusion**), not merely
> *which* cells are present. We test that head-on: a spatial graph neural
> network (reads arrangement) vs. composition-only baselines (XGBoost/MLP on
> cell proportions, blind to arrangement). If the GNN wins, spatial structure
> carries prognostic signal beyond composition.

## Why this framing

This is the result of an evidence-driven search. GNNs applied to spatial-domain
*clustering* (SpaGCN/STAGATE/GraphST) and to *outcome prediction* are both
mature fields, so we do not claim a new method. Following the actual SRI winners
(which won on rigor + interpretability + impact, not novel algorithms), our
contribution is a **rigorous, falsifiable test of a specific biological claim**:
*does spatial arrangement add prognostic signal beyond cell composition?* — a
question most outcome-GNN papers assume rather than test.

## Headline result (synthetic cohort, multi-seed)

The label is set purely by spatial arrangement (immune infiltration vs.
exclusion) with **cell composition held identical across classes**:

| Model (5 seeds) | sees arrangement? | Macro-F1 | AUROC |
|---|---|---|---|
| Logistic regression (composition) | no | 0.34 | 0.58 |
| XGBoost (composition) | no | 0.51 | 0.50 |
| MLP (composition) | no | 0.37 | 0.60 |
| **Spatial GNN (ours)** | **yes** | **1.00** | **1.00** |

**Graph-shuffle ablation:** GNN intact 1.00 → shuffled 0.33 → empty 0.33 ≈
composition baseline (XGBoost 0.51). Destroying the spatial graph erases the
advantage, proving the signal is the *arrangement* itself, not the cells.

(Exact multi-seed numbers in `results/`; the synthetic task is intentionally
controlled — the real-data magnitude is the honest test.)

## Install & run

```bash
cd research/spatial_prognosis
pip install -r requirements.txt        # or reuse ../celiac/venv

python scripts/run_synthetic_demo.py          # full (offline proof)
python scripts/run_synthetic_demo.py --fast    # quick smoke test
```

Real data (imaging mass cytometry with clinical outcomes):

```bash
python scripts/run_real.py             # Jackson-Fischer breast IMC (grade/survival)
```

The recommended cohort is **Jackson-Fischer 2020 breast IMC** (~700 patients,
37 markers, tumor grade + survival; Zenodo `10.5281/zenodo.3518284`). The loader
(`data/real_imc.py`) builds per-patient spatial cell graphs from any long-form
single-cell table + a per-sample clinical label, so MERFISH/CODEX/Xenium
cohorts work the same way.

## The honest test on real data (Jackson-Fischer breast IMC, 372 patients)

On real tumors, cell composition *does* carry prognostic signal, so the
baselines are not at chance. The result is endpoint-dependent and reported as-is:

**Tumor grade (G3 vs G1/G2) — a tumor-cell-intrinsic property:**

| | LogReg | XGBoost | MLP | Spatial GNN |
|---|---|---|---|---|
| Macro-F1 | 0.588 | 0.609 | **0.633** | 0.567 |

The GNN does **not** add over composition — expected, since grade is well
captured by which cells are present.

**5-year survival — where immune *arrangement* (infiltration vs. exclusion) should matter:**

| | LogReg | XGBoost | MLP | Spatial GNN |
|---|---|---|---|---|
| Macro-F1 | 0.445 | 0.503 | 0.445 | **0.544** |
| AUROC | 0.635 | 0.610 | 0.563 | 0.633 |

The GNN has a small macro-F1 edge — **but it fails the falsification test.** The
correct test is *intact vs. shuffled* (both keep neighborhood aggregation; only
the arrangement differs): intact **0.54** ≈ shuffled **0.51**, while only the
*empty* graph drops (0.45). If arrangement carried the signal, shuffling would
collapse it (as it does synthetically, 1.00 → 0.33). It does not — so the small
lift is **generic neighborhood feature-smoothing, not spatial arrangement.**

**Honest read (null result).** On this real cohort, spatial arrangement adds
**no detectable signal beyond cell composition** — for either grade or survival.
This is the opposite of the synthetic benchmark (where arrangement is, by
construction, the only signal). Reported as-is. Results in `results/real_imc_*.json`.

## Project layout

```
spatial_prognosis/
├── README.md
├── requirements.txt
├── docs/proposal.tex            # SRI proposal (health-first)
├── spatial_prognosis/
│   ├── config.py
│   ├── data/
│   │   ├── synthetic.py         # infiltration-vs-exclusion cohort (composition matched)
│   │   └── real_imc.py          # IMC/CODEX/Xenium -> per-patient cell graphs
│   ├── models/gnn.py            # spatial GNN graph classifier (ours)
│   ├── metrics.py               # accuracy, macro-F1, AUROC
│   ├── splits.py                # patient-level splits
│   ├── train.py                 # GNN + composition-baseline trainers
│   ├── experiment.py            # multi-seed comparison
│   └── ablation.py              # graph-shuffle falsification test
├── scripts/
│   ├── run_synthetic_demo.py
│   └── run_real.py
├── results/   └── figures/
```

## Interpretability & impact

Beyond accuracy, the spatial GNN localizes the **cell neighborhoods** that drive
predictions (prognostic niches) — e.g., tumor–immune contact patterns — turning
a prediction into a candidate spatial biomarker. Predictions are research-only
and hypothesis-generating.
