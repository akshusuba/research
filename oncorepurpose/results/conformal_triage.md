# OncoEvidence: split-conformal triage calibration

_2026-06-28T23:31:49.053240+00:00_  ·  mode: full  ·  device: cuda

Wraps the deployment ranker (transductive HeteroGNN on PrimeKG indication edges) in split-conformal prediction so each candidate gets a calibrated confidence and the triage can **abstain** below a coverage target.

## Headline

At target coverage 90%, split-conformal triage achieves empirical coverage 90.3% on held-out true indications (score threshold 0.442), abstaining on 52.9% of a positives+negatives pool and 96.1% of negatives.

## Setup

- Positives split: train=5633, early-stop=939, held-out=2816 (held-out is disjoint from training/early-stopping).
- Calibration positives: 1408; test positives: 1408; test negatives: 1408.
- Held-out raw-score AUROC (context, not the conformal claim): 0.976.
- Nonconformity = 1 - sigmoid link score, calibrated on POSITIVES (coverage is a guarantee about true indications).

## Coverage / abstention by target

| Target coverage | Score threshold | Empirical coverage (test +) | Abstention (pool) | Negative abstention |
|---|---|---|---|---|
| 80% | 0.603 | 80.6% | 58.5% | 97.6% |
| 90% | 0.442 | 90.3% | 52.9% | 96.1% |
| 95% | 0.128 | 94.7% | 47.5% | 89.8% |

## Shortlist calibration

Re-scored 30 shortlist candidates with the deployment model: **25 accepted, 5 abstained** (abstention rate 16.7%) at 90% target coverage. See `repurposing_shortlist_calibrated.json`.

## Honest read & caveats

- Conformal coverage is a *marginal* finite-sample guarantee under exchangeability of the calibration and test positives; both are random holdouts of the same PrimeKG indication edges, so exchangeability is reasonable but the guarantee transfers to *novel* candidates only insofar as they are exchangeable with known indications (they may not be).
- Negatives are sampled (assumed-negative) drug–disease pairs, so the negative-abstention rate is indicative, not a true specificity.
- The calibration is only as good as the ranker; conformal makes the abstention decision *honest*, it does not improve ranking.
