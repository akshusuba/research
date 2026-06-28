# Temporal-split prospective evaluation (full)

_2026-06-28T01:35:46.642060_

## Headline

Prospective (temporal) test on FUTURE oncology indications (cutoff T=2005): GNN AUROC=0.930, structure-blind MLP AUROC=0.882 (graph gain +0.047).

The GNN ranks FUTURE true oncology indications **above chance** above random (drug, cancer) negatives, and the **graph adds value**: the GNN beats the structure-blind control by +0.047 AUROC on this prospective task.

## Setup

- Temporal axis: earliest Europe PMC year co-mentioning each true (drug, cancer) indication pair.
- Cutoff **T = 2005** (p70 of resolved years).
- True oncology indication pairs total: 1629; sampled 350; years resolved 341.
- **PAST** (year ≤ T, in message-passing graph + train): 240; **FUTURE** (year > T, held-out prospective test): 101.
- FUTURE target edges removed from the graph (no leakage); all drug↔disease therapeutic edges stripped exactly as in the inductive split.
- Test negatives: random (drug, oncology-cancer) pairs at 5.0× positives; 3 seed(s).

## Results (mean ± std over seeds)

| Model | AUROC | AUPRC | recall@50 | recall@100 | recall@200 |
|---|---|---|---|---|---|
| **GNN (graph)** | 0.930 ± 0.005 | 0.707 ± 0.039 | 0.393 ± 0.025 | 0.703 ± 0.014 | 0.927 ± 0.033 |
| MLP (structure-blind) | 0.882 ± 0.020 | 0.592 ± 0.032 | 0.337 ± 0.016 | 0.587 ± 0.012 | 0.822 ± 0.032 |

## Honest read

- Prospective AUROC (GNN) = **0.930** vs 0.5 chance.
- Graph vs structure-blind control: **+0.047** AUROC.
- Caveats: first-evidence year is an *approximate* proxy (earliest literature co-mention, not regulatory/discovery date) and is noisy; the sample is a seeded subset of oncology pairs; Europe PMC co-mention can precede or lag true establishment of an indication. Treat magnitudes as indicative, not exact.
