# ClinicalTrials.gov Validation of Repurposing Predictions

**Question:** Do the model's top-ranked *novel* drug->cancer predictions (pairs not already known indications in PrimeKG) appear as real interventional oncology trials more often than matched control pairs? ClinicalTrials.gov is fully independent of the knowledge graph and the model.

**Hit definition:** >=1 interventional ClinicalTrials.gov study for query.intr=drug & query.cond=cancer.

## Headline

| set | hits / n | fraction with a trial |
| --- | --- | --- |
| top | 6 / 108 | 5.6% |
| low | 4 / 108 | 3.7% |
| random | 9 / 108 | 8.3% |

**top_vs_random:** 5.6% vs 8.3% (ratio **0.67x**, odds ratio 0.65, Fisher one-sided **p = 0.858**).
**top_vs_low:** 5.6% vs 3.7% (ratio **1.50x**, odds ratio 1.53, Fisher one-sided **p = 0.374**).

**AUROC** of "interventional trial exists" vs model score over 324 scored novel pairs: **0.765** (0.5 = no signal).

## Concrete corroborated novel predictions (top set, real trial exists)

| drug | cancer | model score | rank | # trials | example NCT |
| --- | --- | --- | --- | --- | --- |
| (R)-Bicalutamide | prostate cancer | 0.963 | 2 | 1 | NCT00666666 |
| 3-Tyrosine | melanoma | 0.957 | 2 | 14 | NCT02465060 |
| Phenothiazine | breast carcinoma | 0.925 | 4 | 9 | NCT01298193 |
| Phenothiazine | gastric cancer | 0.921 | 3 | 2 | NCT05232357 |
| Emapalumab | non-small cell lung carcinoma | 0.917 | 1 | 1 | NCT06439914 |
| Phenothiazine | colorectal cancer | 0.915 | 2 | 3 | NCT04842968 |

## Caveats

- **Name matching is fuzzy.** Drug/cancer strings are passed to ClinicalTrials.gov's free-text + synonym search; some true matches may be missed and some loose matches counted.
- **A registered trial is not evidence of efficacy.** It shows someone judged the drug-cancer pair worth testing in humans, which is exactly the orthogonal plausibility signal sought here.
- **Sample sizes are modest**; treat p-values and the AUROC as indicative, not definitive.
- **Reverse-causality risk is low:** ClinicalTrials.gov is not an input to PrimeKG or the model, so enrichment cannot be an artifact of training leakage, but popular/older drugs are over-represented in both trials and predictions.