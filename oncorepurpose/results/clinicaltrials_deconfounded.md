# ClinicalTrials.gov validation, popularity-deconfounded (full)

_2026-06-28T20:57:59.907886_

## Why this test

The raw-score orthogonal check (see `clinicaltrials_validation.md`) is confounded by popularity on two sides: some drugs are trialed for every cancer, and some cancers are trialed with every drug. A positive raw AUROC can therefore reflect 'popular drug' or 'popular cancer' rather than 'right drug for this cancer'. We remove **both** effects and test only what is left: the drug x cancer **interaction**.

## Headline -- interaction test (primary)

We fit an additive two-way model score(drug, cancer) = mu + drug-effect + cancer-effect on the evaluated pairs and take the residual. The residual is how much the model elevates a *specific* pair beyond what the drug's and the cancer's overall popularity predict. We then ask whether that residual predicts a real interventional trial.

- **Interaction AUROC = 0.475** (0.5 = no signal beyond popularity), structure-aware permutation p = **0.853**.
- Evaluated over **1843** novel (drug, cancer) pairs spanning **103** focus drugs x **18** cancers (overall pair hit-fraction 0.184).

**Read:** no clean interaction signal once both popularity effects are removed. Once both popularity baselines are subtracted, the model's per-pair scores no longer track which specific drug-cancer combinations have trials. In other words, the raw deployment scores carry drug- and cancer-level popularity, but not pairwise repurposing specificity that this external registry can corroborate. We report this as a characterized limitation rather than a win, consistent with the project's honest-evaluation stance; the time-split prospective analysis (Finding 5) is where specific predictive signal does show up.

## Supporting -- within-drug stratified AUROC (controls drug popularity only)

- **Within-drug AUROC = 0.821**, permutation p = **0.0005**, from **49** drugs over **2649** within-drug cancer pairs.
- Top-cancer sign test: for **31/49** drugs the single highest-scored novel cancer has a real trial, vs a per-drug base rate of 0.391 (binomial p = 0.000541).
(This one controls drug popularity but not cancer popularity, which is why the interaction test above is the cleaner number.)

## Clean interaction wins (trial pairs the model elevated most above BOTH baselines)

| drug | cancer | model score | interaction residual | example NCT |
|---|---|---|---|---|
| Dexamethasone | bilineal acute myeloid leukemia | 0.432 | +0.0072 | NCT02135874 |
| Folic acid | gallbladder carcinoma | 0.789 | +0.0048 | NCT01572324 |
| Tromethamine | cervical cancer | 0.998 | +0.0025 | NCT04246697 |
| Tromethamine | melanoma | 0.996 | +0.0025 | NCT03715985 |
| Tromethamine | prostate cancer | 0.997 | +0.0025 | NCT07557901 |
| Tromethamine | ovarian cancer | 0.997 | +0.0024 | NCT02641639 |
| Ephedrine | cervical cancer | 0.998 | +0.0023 | NCT05874492 |
| Ephedrine | prostate cancer | 0.997 | +0.0023 | NCT07626151 |
| Norepinephrine | hepatocellular carcinoma | 0.997 | +0.0023 | NCT02472249 |
| Norepinephrine | cervical cancer | 0.998 | +0.0023 | NCT01418118 |
| Norepinephrine | melanoma | 0.996 | +0.0023 | NCT03347123 |
| Tromethamine | colorectal cancer | 0.980 | +0.0023 | NCT02509143 |
| Tromethamine | gastric cancer | 0.982 | +0.0023 | NCT02650375 |
| Tromethamine | non-small cell lung carcinoma | 0.981 | +0.0023 | NCT03715985 |
| Tromethamine | breast carcinoma | 0.983 | +0.0023 | NCT06150898 |

## Caveats

- Trial existence is a plausibility signal, not efficacy.
- Name matching uses ClinicalTrials.gov free-text + synonym search (some misses / loose matches).
- The focus-drug pool is the model's own top-scored novel candidates, so this tests whether, among drugs the model likes, it points at the right cancer; it does not re-test candidate selection itself.
- The interaction residual is an additive de-confounder; strong multiplicative popularity effects could leave minor residual structure, so treat the magnitude as indicative.
