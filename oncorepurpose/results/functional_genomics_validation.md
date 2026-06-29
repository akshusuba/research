# OncoEvidence — Functional-Genomics Validation of Mechanism Paths

> **OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided Cancer Drug Repurposing.**

Independent, non-knowledge-graph evidence that the extracted mechanism paths are biologically real, used to attack the project's weakest result — the mechanism-separation AUROC that drops to **~0.609** against *shared-target* hard negatives (vs ~0.887 vs random). All signals below are computed from public functional-genomics data (DepMap CRISPR, GTEx) with **no reference to PrimeKG or the literature**.

- True oncology indication pairs: **400**
- Shared-target hard negatives: **368**
- Disease→DepMap-lineage coverage: **89.6%**

## Phase 1 — DepMap CRISPR (Chronos) dependency

Per pair, the mean CRISPR gene-effect of the mechanism bridge/target gene(s) in the matching cancer **lineage's** cell lines (more negative = stronger dependency). Tests whether TRUE-MOA genes are stronger dependencies than shared-target decoy genes.

| Gene aggregation | TRUE mean Chronos | NEG mean Chronos | AUROC (TRUE vs shared-target) | Mann-Whitney p |
|---|---|---|---|---|
| mean_dependency | -0.304 | -0.186 | **0.607** | 4.61e-06 |
| strongest_dependency | -1.281 | -1.027 | **0.583** | 3.71e-04 |
| best_path_dependency | -0.279 | -0.181 | **0.553** | 2.29e-02 |

DepMap genes scored: 1367. Independent of the graph and the literature.

## Phase 2 — GTEx target tissue expression context

Median log1p(TPM) of bridge genes in the lineage's matched normal tissue. TRUE mean **2.569** vs NEG **2.074** (standalone AUROC 0.618, p=1.86e-06).

## Phase 3 — LINCS L1000 connectivity

_clue.io API key not set (CLUE_API_KEY / LINCS_API_KEY); L1000 connectivity requires an authenticated key. SKIPPED per plan -- not burning time here._

## Phase 4 — Hard-negative-aware specificity classifier

Cross-validated (5-fold) separation of TRUE pairs from SHARED-TARGET hard negatives using [structure-only mechanism_score, DepMap dependency, GTEx expression, target promiscuity, direct-target flag]. The question: do functional-genomics features beat the path-only ~0.609 baseline?

| Model / feature set | CV AUROC | Δ vs path-only |
|---|---|---|
| path_only_baseline | 0.609 | — |
| lr_structure_only | 0.630 | +0.021 |
| lr_struct_plus_depmap | 0.663 | +0.054 |
| lr_struct_plus_gtex | 0.640 | +0.031 |
| lr_functional_only | 0.624 | +0.015 |
| lr_all_features | 0.668 | +0.059 |
| gbm_structure_only | 0.731 | +0.121 |
| gbm_all_features | 0.751 | +0.141 |

Best model **gbm_all_features** = **0.751**, which **beats** the path-only baseline (0.609) by **+0.141**.

## Verdict — does this move the needle?

**Yes — modestly but genuinely, with independent evidence.** DepMap CRISPR dependency *alone* separates TRUE from shared-target negatives at AUROC **0.607** (Mann-Whitney p=4.6e-06), and GTEx target expression at **0.618** — both computed with **zero** reference to PrimeKG or the literature, so they are truly orthogonal corroboration that the bridge genes are biologically real cancer dependencies, not graph artifacts.

On the actual hard task (separating TRUE pairs from shared-target decoys), adding the independent functional features lifts the path-only baseline from **0.609**: a clean, interpretable logistic model with DepMap reaches **0.663** (Δ +0.054), and the gradient-boosted model with all features reaches **0.751** (Δ +0.141).

**Honesty control.** A GBM on *structure-only* features already reaches **0.731** (most of the GBM gain is the model capturing non-linear structure, e.g. direct-target × promiscuity interactions). Adding the functional-genomics features takes the GBM from 0.731 → 0.751 (Δ +0.020), and adds Δ +0.054 to the linear model — so DepMap/GTEx contribute a real, reproducible increment on top of structure, even if part of the headline GBM jump is model capacity rather than new biology.

**Caveats.**
- Coarse keyword disease→lineage mapping (89.6% coverage); subtype-level lineage matching would be more precise.
- Per-pair dependency aggregates over *all* extracted bridge genes (mean Chronos); diluting over many genes weakens signal — the single-best-path variant is weaker (AUROC ~0.55), so the signal is distributed, not driven by one gene.
- Shared-target decoys can hit the *same* protein as the true drug; those pairs are intrinsically inseparable by a target-gene dependency, which caps the achievable AUROC.
- DepMap measures dependency in *cell lines*, GTEx expression in *normal* tissue; neither is the patient tumour. LINCS L1000 was skipped (no clue.io key).
- n=768 pairs; the GBM gain is stable across 5 CV seeds (std ≤0.005) but should be read with the structure-only control above.

