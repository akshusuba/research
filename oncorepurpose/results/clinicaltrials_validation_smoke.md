# ClinicalTrials.gov Validation (SMOKE)

**Question:** Do the model's top-ranked *novel* drug->cancer predictions (pairs not already known indications in PrimeKG) appear as real interventional oncology trials more often than matched control pairs? ClinicalTrials.gov is fully independent of the knowledge graph and the model.

**Hit definition:** >=1 interventional ClinicalTrials.gov study for query.intr=drug & query.cond=cancer.

## Headline

| set | hits / n | fraction with a trial | unique drugs |
| --- | --- | --- | --- |
| top | 1 / 6 | 16.7% | 2 |
| random | 1 / 6 | 16.7% | 6 |

- **top_vs_random:** 16.7% vs 16.7% (ratio **1.00x**, odds ratio 1.00, Fisher one-sided **p = 0.773**).

**AUROC** of "interventional trial exists" vs raw model score over 6 scored novel pairs (top+low+random): **1.000**, Mann-Whitney p = 0.167 (0.5 = no signal).

## Popularity / promiscuity confound

How concentrated are the trial "hits" on a few broadly-indicated drugs? If one drug accounts for most hits, a positive raw-score AUROC reflects drug popularity (such drugs are scored high *everywhere* and trialed *everywhere*) rather than specific repurposing insight.

| set | hits | distinct hit-drugs | most frequent hit-drug (share) |
| --- | --- | --- | --- |
| top | 1 | 1 | Cyclosporine (1/1, 100%) |
| random | 1 | 1 | Tremelimumab (1/1, 100%) |
| top_lift | 0 | 0 | - |

## Concrete corroborated novel predictions -- raw-score top set (one row per distinct drug)

| drug | cancer | model score | rank | # trials | example NCT |
| --- | --- | --- | --- | --- | --- |
| Cyclosporine | metastatic melanoma | 0.996 |  | 2 | NCT00006233 |

## Caveats

- **Name matching is fuzzy.** Drug/cancer strings are passed to ClinicalTrials.gov's free-text + synonym search; some true matches may be missed and some loose matches counted.
- **A registered trial is not evidence of efficacy.** It shows someone judged the drug-cancer pair worth testing in humans, which is exactly the orthogonal plausibility signal sought here.
- **Sample sizes are modest**; treat p-values and the AUROC as indicative, not definitive.
- **Reverse-causality risk is low:** ClinicalTrials.gov is not an input to PrimeKG or the model, so enrichment cannot be an artifact of training leakage, but popular/older drugs are over-represented in both trials and predictions.