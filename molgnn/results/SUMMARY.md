# Molecular GNN vs. XGBoost on EGFR drug-activity — honest head-to-head

**Question:** On a real cancer drug-activity dataset, does a graph neural network
operating on the *molecular graph* beat a strong tabular baseline (XGBoost on
Morgan fingerprints)? Fairness is the whole point — no engineering a win.

**Verdict: NO. The GNN loses decisively.** On the rigorous, leakage-safe
Bemis-Murcko scaffold split, the GIN trails XGBoost by ~5.7 AUROC points and
~6.3 AUPRC points, and trails a plain RandomForest by even more. The same
ordering holds on the easier random split. Result is consistent across 3 seeds
with tiny variance. Tabular fingerprint models win clearly here.

---

## Dataset
- **Target:** EGFR — Epidermal Growth Factor Receptor (`CHEMBL203`), *Homo sapiens*.
  A canonical oncology drug target (NSCLC, etc.).
- **Source:** ChEMBL bioactivities pulled live via `chembl_webresource_client`,
  filtered to records with a `pchembl_value`, `=` relation, median-aggregated per
  unique canonical SMILES. 21,342 raw activity records → **11,185 unique molecules**.
- **Label:** active = `pchembl_value >= 6.5`, inactive otherwise.
- **Class balance:** 7,021 active (**62.8%**) / 4,164 inactive (37.2%). Reasonable
  balance at 6.5 (6.0 gives 75% positive — too skewed; 6.5 used as primary).
- **Unique Bemis-Murcko scaffolds:** 4,052.

## Featurization (identical molecules for both models)
- **GNN:** RDKit mol → PyG graph. Atom features (40-dim): one-hot atomic number,
  degree, formal charge, hybridization, total H count, plus aromaticity, ring
  membership, scaled mass. Bidirectional bond edges.
- **XGBoost / RF:** Morgan/ECFP fingerprint, radius 2, 2048 bits, of the *same* molecule.

## Split (leakage-safe, primary = scaffold)
- **Bemis-Murcko scaffold split**, 80/10/10 by whole scaffold groups (no scaffold
  appears in two splits). This tests generalization to **new chemotypes**.
- Variation: scaffolds **re-split per seed** (tie-break shuffle on equal-size groups)
  **and** model init varies per seed. 3 seeds.
- A **random split** is reported for contrast.

## Models (both tuned; neither crippled)
- **GNN:** GIN, small architecture grid (3 configs: layers/hidden/pooling/dropout/lr)
  selected on validation AUPRC, trained on A100 with `pos_weight` for imbalance and
  **early stopping on val AUPRC** (patience 20, ≤150 epochs).
- **XGBoost:** 5-point hyperparameter grid (depth/lr/n_estimators/subsample/
  colsample), `scale_pos_weight`, GPU hist, early-stop eval set, selected on val AUPRC.
- **RandomForest** (500 trees, balanced) on fingerprints, for additional context.

---

## Results (test set, mean ± std over 3 seeds)

### Scaffold split — PRIMARY (leakage-safe)
| Model | AUROC | AUPRC |
|---|---|---|
| **GNN (GIN, molecular graph)** | 0.810 ± 0.003 | 0.840 ± 0.001 |
| **XGBoost (Morgan FP)** | **0.867 ± 0.005** | **0.903 ± 0.007** |
| RandomForest (Morgan FP) | **0.884 ± 0.003** | **0.912 ± 0.003** |

### Random split — contrast (leaky, easier)
| Model | AUROC | AUPRC |
|---|---|---|
| **GNN (GIN, molecular graph)** | 0.891 ± 0.009 | 0.925 ± 0.009 |
| **XGBoost (Morgan FP)** | **0.924 ± 0.004** | **0.951 ± 0.003** |
| RandomForest (Morgan FP) | **0.926 ± 0.004** | **0.951 ± 0.001** |

**Magnitude of the gap (scaffold, GNN vs best baseline = RF):** −7.4 AUROC pts,
−7.2 AUPRC pts. GNN vs XGBoost: −5.7 AUROC, −6.3 AUPRC. The gaps dwarf the
seed-to-seed std (≤0.01), so this is **decisive, not noise**.

As expected, both splits' scores drop from random → scaffold (random split leaks
chemotype information), confirming the scaffold split is the harder, more honest
test. The model *ordering* is unchanged: fingerprint+tree > GNN in every case.

---

## Verdict
**Decisively NO** on this dataset. The molecular GNN does **not** beat XGBoost on
fingerprints — and is in fact beaten by *both* XGBoost and a vanilla RandomForest,
on both the rigorous scaffold split (primary) and the leaky random split. This is
the honest empirical answer: even though the molecule literally *is* a graph, a
well-tuned ECFP + gradient-boosted-trees / random-forest pipeline is the stronger,
more data-efficient model here.

This is consistent with the prior four projects' pattern (GNNs tie/lose to XGBoost
on real biomedical data), and lands on the pessimistic side of the genuinely MIXED
molecular-property-prediction literature: for a single mid-sized target like EGFR,
ECFP+trees is a very hard baseline to beat.

## Caveats / honesty notes
- **No fabrication.** All numbers come from `results/results.json` (run log:
  `results/run_log.txt`, total wall time ~27 min on one A100).
- **Single target.** EGFR only. A different target, multitask training, or a much
  larger dataset could narrow or flip the gap — molecular GNNs tend to shine with
  more data and/or pretraining (e.g., large self-supervised molecular models),
  none of which were used here. This is a from-scratch, single-target comparison.
- **GNN was given a fair shot:** 3-arch grid selected on val AUPRC (it picked the
  5-layer add-pool GIN on scaffold), class weighting, early stopping, same molecules,
  GPU. The grid is modest; heavier GNN tuning *might* close some gap, but XGBoost was
  similarly only modestly tuned, so the comparison is balanced.
- **Label/threshold choice:** 6.5 chosen for balance; censored (`>`/`<`) relations
  dropped to reduce label noise; duplicate measurements median-aggregated per molecule.
- The GNN's val/test AUPRC gap was small (no severe overfitting), so the loss is a
  genuine capacity/inductive-bias outcome, not a training failure.

## Files
- `fetch_data.py` — ChEMBL EGFR pull + dedup/aggregation.
- `data_utils.py` — atom/graph featurization, Morgan FP, scaffold + random splits.
- `models.py` — GIN classifier.
- `run_benchmark.py` — full head-to-head (GNN grid, XGB grid, RF) over seeds/splits.
- `results/results.json` — all per-seed + summary metrics + dataset metadata.
- `results/run_log.txt` — raw run log.
