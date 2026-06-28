# ClinicalTrials.gov Validation of Repurposing Predictions

**Question:** Do the model's top-ranked *novel* drug->cancer predictions (pairs not already known indications in PrimeKG) appear as real interventional oncology trials more often than matched control pairs? ClinicalTrials.gov is fully independent of the knowledge graph and the model.

**Hit definition:** >=1 interventional ClinicalTrials.gov study for query.intr=drug & query.cond=cancer.

**Sets.** `top` = top-k novel pairs per cancer ranked by **raw model score** (the primary test of "top-ranked predictions"). `low` = bottom-band novel pairs by raw score (matched). `random` = random drug x cancer pairs over the same cancers, known indications excluded. `top_lift` = top-k by **specificity lift** -- the ranking actually used for the deployed shortlist (shown for an honest contrast).

## Headline

| set | hits / n | fraction with a trial | unique drugs |
| --- | --- | --- | --- |
| top | 18 / 270 | 6.7% | 29 |
| low | 12 / 270 | 4.4% | 114 |
| random | 20 / 270 | 7.4% | 265 |
| top_lift | 11 / 270 | 4.1% | 132 |

- **top_vs_random:** 6.7% vs 7.4% (ratio **0.90x**, odds ratio 0.89, Fisher one-sided **p = 0.693**).
- **top_vs_low:** 6.7% vs 4.4% (ratio **1.50x**, odds ratio 1.54, Fisher one-sided **p = 0.174**).
- **top_lift_vs_random:** 4.1% vs 7.4% (ratio **0.55x**, odds ratio 0.53, Fisher one-sided **p = 0.969**).
- **top_lift_vs_low:** 4.1% vs 4.4% (ratio **0.92x**, odds ratio 0.91, Fisher one-sided **p = 0.665**).

**AUROC** of "interventional trial exists" vs raw model score over 810 scored novel pairs (top+low+random): **0.676**, Mann-Whitney p = 1.52e-05 (0.5 = no signal).

## Popularity / promiscuity confound

How concentrated are the trial "hits" on a few broadly-indicated drugs? If one drug accounts for most hits, a positive raw-score AUROC reflects drug popularity (such drugs are scored high *everywhere* and trialed *everywhere*) rather than specific repurposing insight.

| set | hits | distinct hit-drugs | most frequent hit-drug (share) |
| --- | --- | --- | --- |
| top | 18 | 3 | Folic acid (14/18, 78%) |
| random | 20 | 20 | Lorvotuzumab mertansine (1/20, 5%) |
| top_lift | 11 | 8 | Phenothiazine (3/11, 27%) |

## Interpretation

Top raw-score novel predictions are NOT enriched vs random (0.90x, p=0.693). Raw model score does rank trial-existence above chance (AUROC 0.676, p=1.5e-05): higher-scored novel pairs are more likely to have a real interventional trial. But this is heavily confounded by drug popularity: a single broadly-indicated drug (Folic acid) accounts for 78% of the top set's hits -- such drugs score high for every cancer and are trialed for every cancer. Consistently, the specificity-lift ranking (the deployed shortlist ordering, which de-confounds popularity) shows NO trial enrichment (0.55x vs random, p=0.969): the genuinely specific novel predictions are not (yet) over-represented in trials. Bottom line: weak and confounded independent signal -- this corroborates that the model's raw scores track human trial activity (mostly via popular drugs), but does NOT yet provide clean real-world validation of the specific novel repurposing shortlist.

## Concrete corroborated novel predictions -- raw-score top set (one row per distinct drug)

| drug | cancer | model score | rank | # trials | example NCT |
| --- | --- | --- | --- | --- | --- |
| Folic acid | cervical cancer | 0.998 | 8 | 95 | NCT03110926 |
| Neostigmine | breast carcinoma | 0.983 | 6 | 1 | NCT02839668 |
| Dexamethasone | gallbladder carcinoma | 0.784 | 14 | 1 | NCT00016380 |

## Concrete corroborated novel predictions -- specificity-lift shortlist ranking (one row per distinct drug)

| drug | cancer | model score | rank | # trials | example NCT |
| --- | --- | --- | --- | --- | --- |
| (R)-Bicalutamide | prostate cancer | 0.963 | 2 | 1 | NCT00666666 |
| Compound 4-D | hepatocellular carcinoma | 0.961 | 7 | 1 | NCT05903456 |
| 3-Tyrosine | melanoma | 0.957 | 2 | 14 | NCT02465060 |
| Phenothiazine | breast carcinoma | 0.925 | 4 | 9 | NCT01298193 |
| Nonoxynol-9 | colorectal cancer | 0.919 | 10 | 1 | NCT01200303 |
| Emapalumab | non-small cell lung carcinoma | 0.917 | 1 | 1 | NCT06439914 |
| Racepinephrine | neuroblastoma | 0.901 | 9 | 5 | NCT06607692 |
| Dexamethasone | bilineal acute myeloid leukemia | 0.432 | 10 | 1 | NCT02135874 |

## Caveats

- **Name matching is fuzzy.** Drug/cancer strings are passed to ClinicalTrials.gov's free-text + synonym search; some true matches may be missed and some loose matches counted.
- **A registered trial is not evidence of efficacy.** It shows someone judged the drug-cancer pair worth testing in humans, which is exactly the orthogonal plausibility signal sought here.
- **Sample sizes are modest**; treat p-values and the AUROC as indicative, not definitive.
- **Reverse-causality risk is low:** ClinicalTrials.gov is not an input to PrimeKG or the model, so enrichment cannot be an artifact of training leakage, but popular/older drugs are over-represented in both trials and predictions.