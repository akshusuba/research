# Counterfactual Edge-Faithfulness of GNN Mechanism Predictions

**Verdict: FAITHFUL**

Does the joint GNN, when it names the curated DrugMechDB bridge gene for a held-out
(drug, cancer) pair, actually rely on that gene's curated mechanism-of-action (MOA)
`drug_protein` edge -- or on spurious structure? We test this with a counterfactual
edge-removal experiment on the trained model (no retraining): recompute embeddings
after deleting edges and measure the bridge gene's mechanism score/rank drop.

- Setup: inductive cold-disease (oncology) split; joint GNN (link BCE + InfoNCE
  mechanism aux), seeds [0, 1], 50 epochs, 3
  random-removal draws per pair.
- Sample: **316** held-out (drug, cancer, bridge-gene) instances
  (covered pairs whose curated bridge gene is an actual `drug_protein` edge).

## Headline faithfulness contrast (mechanism score, logits)

| Condition | Mean score drop (seed mean±std) |
|---|---|
| REMOVE-MOA (this pair's MOA edge) | 0.124 ± 0.001 |
| REMOVE-RANDOM (matched other targets of same drug) | -0.001 ± 0.004 |
| **Contrast (MOA − random)** | **0.125 ± 0.003** |

- Ratio of mean drops (MOA / random): **n/a (random drop ≈ 0)**
- Fraction of instances where MOA removal hurts more than random: **0.833 ± 0.019**
- Paired Wilcoxon (MOA drop > random drop), pooled: **p = 3.62e-34**
- Separation AUROC (MOA vs random drops): **0.835**
- Mean rank degradation (positions): MOA = 209.540 ± 26.298 vs random = 2.046 ± 1.437
- Sufficiency (keep only MOA edges) score drop: -0.008 ± 0.042 (smaller = more sufficient)

## Interpretation

FAITHFUL: MOA-edge removal degrades the bridge gene's score significantly more than removing other target edges of the same drug. Pooled over 316 held-out (drug,cancer,bridge-gene) instances: mean score drop under MOA removal = 0.124 vs random removal = -0.001 (contrast = 0.124); fraction of instances where MOA removal hurts more = 0.832; paired Wilcoxon p (MOA>random) = 3.62e-34; separation AUROC = 0.835. Mean rank degradation: MOA = 211.0 vs random = 2.1 positions.

### Caveats
- "Score" is the mechanism-head logit; drops are in logit space (monotonic, not
  probability). Ranks are over all `gene_protein` nodes.
- Faithfulness is measured only on covered pairs whose bridge gene is a real
  `drug_protein` edge (others have no MOA edge to ablate); this is the population
  where the question is well-posed, not all predictions.
- Removing an edge changes both endpoints' embeddings; the random control isolates
  "is it THIS edge" by deleting the same drug's other target edges (with degree-
  matched global top-up when a drug has too few other targets).
- Small held-out covered set; treat as a faithfulness probe, not a population estimate.
