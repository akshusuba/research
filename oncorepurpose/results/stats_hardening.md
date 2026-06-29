# OncoEvidence: statistical hardening

_2026-06-28T23:33:09.119595+00:00_  ·  mode: full  ·  bootstrap draws: 2000

## Headline

Mechanism separation AUROC 0.879 (95% CI 0.856-0.900). Prospective GNN AUROC stays above chance at all cutoffs (yes); graph gain over MLP [T=2000:+0.065, T=2005:+0.037, T=2010:+0.029].

## A. Bootstrap 95% CIs (AUROC)

Percentile CIs from resampling the scored pairs (positives and negatives independently, with replacement).

| Comparison | AUROC | 95% CI | n_pos | n_neg |
|---|---|---|---|---|
| Mechanism separation (true vs random) | 0.879 | [0.856, 0.900] | 400 | 400 |
| Hard-negative: random | 0.887 | [0.865, 0.908] | 400 | 400 |
| Hard-negative: degree_matched | 0.743 | [0.708, 0.777] | 400 | 400 |
| Hard-negative: oncology_drug | 0.870 | [0.844, 0.892] | 400 | 400 |
| Hard-negative: shared_target | 0.609 | [0.569, 0.649] | 400 | 368 |

## B. Multi-cutoff prospective temporal split

Earliest-evidence years reused from the Europe PMC cache (no network). FUTURE indication edges (year > T) are held out of the message-passing graph; the model must rank them above sampled negatives using only PAST structure. GNN (graph) vs FeatureMLP (structure-blind).

| Cutoff T | PAST train + | FUTURE test + | GNN AUROC | MLP AUROC | Graph gain |
|---|---|---|---|---|---|
| 2000 | 183 | 126 | 0.930 ± 0.006 | 0.865 ± 0.009 | +0.065 |
| 2005 | 204 | 101 | 0.933 ± 0.001 | 0.896 ± 0.005 | +0.037 |
| 2010 | 245 | 53 | 0.935 ± 0.004 | 0.907 ± 0.009 | +0.029 |

- GNN AUROC trend across cutoffs: slope **+0.0005/yr**; above chance at every cutoff: **True**.
- Graph-gain trend (GNN−MLP): slope **-0.0037/yr**; positive at every cutoff: **True**.

## Honest read & caveats

- A bootstrap CI captures sampling variability of the *scored pairs*, not the upstream choices (pair sampling, negative construction); the shared_target CI sitting near 0.5–0.6 confirms the mechanism signal is weak against same-target decoys, exactly as the point estimate warned.
- The temporal axis is an *approximate* first-evidence proxy (earliest Europe PMC co-mention), so absolute AUROCs are indicative; the value here is the consistency of the GNN-over-MLP gap across multiple cutoffs.
- Fewer FUTURE positives at later cutoffs widen the effective uncertainty; trends across only three cutoffs are directional, not a fitted law.
