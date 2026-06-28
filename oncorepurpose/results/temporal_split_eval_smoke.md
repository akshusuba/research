# Temporal-split prospective evaluation (smoke)

_2026-06-28T01:27:10.657155_

## Headline

Prospective (temporal) test on FUTURE oncology indications (cutoff T=2015): GNN AUROC=0.980, structure-blind MLP AUROC=0.840 (graph gain +0.140).

The GNN ranks FUTURE true oncology indications **above chance** above random (drug, cancer) negatives, and the **graph adds value**: the GNN beats the structure-blind control by +0.140 AUROC on this prospective task.

## Setup

- Temporal axis: earliest Europe PMC year co-mentioning each true (drug, cancer) indication pair.
- Cutoff **T = 2015** (p70 of resolved years).
- True oncology indication pairs total: 1629; sampled 16; years resolved 16.
- **PAST** (year ≤ T, in message-passing graph + train): 11; **FUTURE** (year > T, held-out prospective test): 5.
- FUTURE target edges removed from the graph (no leakage); all drug↔disease therapeutic edges stripped exactly as in the inductive split.
- Test negatives: random (drug, oncology-cancer) pairs at 2.0× positives; 1 seed(s).

## Results (mean ± std over seeds)

| Model | AUROC | AUPRC | recall@5 | recall@10 |
|---|---|---|---|---|
| **GNN (graph)** | 0.980 ± 0.000 | 0.967 ± 0.000 | 0.800 ± 0.000 | 1.000 ± 0.000 |
| MLP (structure-blind) | 0.840 ± 0.000 | 0.811 ± 0.000 | 0.600 ± 0.000 | 1.000 ± 0.000 |

## Honest read

- Prospective AUROC (GNN) = **0.980** vs 0.5 chance.
- Graph vs structure-blind control: **+0.140** AUROC.
- Caveats: first-evidence year is an *approximate* proxy (earliest literature co-mention, not regulatory/discovery date) and is noisy; the sample is a seeded subset of oncology pairs; Europe PMC co-mention can precede or lag true establishment of an indication. Treat magnitudes as indicative, not exact.
