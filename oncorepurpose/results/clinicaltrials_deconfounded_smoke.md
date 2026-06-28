# ClinicalTrials.gov validation, popularity-deconfounded (smoke)

_2026-06-28T20:41:25.664088_

## Why this test

The raw-score orthogonal check is confounded by drug popularity: broadly-trialed drugs score high for every cancer and are trialed for every cancer, so a positive raw AUROC can reflect 'popular drug' rather than 'right drug for this cancer'. Here we **condition on the drug**: every comparison is between two cancers for the *same* drug, so the drug's overall popularity cancels out.

## Headline

- **Within-drug stratified AUROC = 0.878** (0.5 = no signal), permutation p = **0.00399**.
- Built from **3** drugs that have both a trial-cancer and a non-trial-cancer, over **90** comparable within-drug cancer pairs.
- Top-cancer sign test: for **1/3** drugs, the model's single highest-scored novel cancer is one with a real interventional trial, versus a per-drug base rate of 0.296 (binomial p = 0.652).
- Pairs evaluated: 108 over 6 focus drugs x 18 cancers (overall pair hit-fraction 0.148).

**Read:** conditioning on the drug, the model shows a real, popularity-free signal: it tends to score the cancer a drug is genuinely being trialed for above the same drug's other cancers, which drug popularity alone cannot explain.

## Clean within-drug wins (top-scored novel cancer has a trial; large gap over non-trial cancers)

| drug | model's top novel cancer | score | gap over best non-trial cancer | trial cancers / total | example NCT |
|---|---|---|---|---|---|
| Folic acid | cervical cancer | 0.998 | +0.028 | 14/18 | NCT03110926 |

## Caveats

- Trial existence is a plausibility signal, not efficacy.
- Name matching uses ClinicalTrials.gov free-text + synonym search (some misses / loose matches).
- The focus-drug pool is the model's own top-scored novel candidates, so this tests whether, among drugs the model likes, it points at the right cancer; it does not re-test candidate selection itself.
- Within-drug strata with only one class contribute nothing to the AUROC (correctly).
