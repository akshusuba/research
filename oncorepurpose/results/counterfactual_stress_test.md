# OncoEvidence — Counterfactual Mechanism Stress Test

> **OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided Cancer Drug Repurposing.** The contribution is the *evaluation*: we test whether the proposed mechanism path actually matters — is it causally important, biologically specific, and literature-supported? — rather than whether we beat a ranker.

**4/4 counterfactual tests pass. VERDICT: MOVES THE NEEDLE — the model demonstrably relies on the true MOA edge and rejects fake mechanisms on the tests that probe causal importance and specificity.**

- Config: seeds [0], 40 epochs, ≤200 held-out pairs/seed (T1/T2), 30 verifier pairs (T3), 400 true pairs (T4); device cuda; DrugMechDB drugs mapped: 1062.

## Test 1 — Target-edge ablation (causal importance)

Remove the curated drug→target MOA edge vs a matched random `drug_protein` edge of the same drug, then measure the bridge gene's mechanism-score drop.

| Condition | Mean mechanism-score drop |
|---|---|
| REMOVE-MOA | 0.165 |
| REMOVE-RANDOM (matched) | -0.003 |
| **Contrast (MOA − random)** | **0.168** |

- Fraction faithful (MOA hurts more than random): **0.872** over 337 instances
- Paired Wilcoxon (MOA drop > random drop): **p = 1.05e-40**
- Separation AUROC (MOA vs random drops): **0.878**
- Rank degradation: MOA = 308.7 vs random = 0.1 positions

## Test 2 — Wrong-target substitution (specificity)

Replace the true target with a drug-degree-matched decoy gene that is NOT associated with the disease, and score the fake mechanism through the trained mechanism head.

- **Rejection rate** (true MOA scored above decoy): **0.926** over 337 instances
- **True-vs-decoy AUROC**: **0.923**
- Mean mechanism score: true = 3.775 vs decoy = -1.959 (margin 5.734)

## Test 3 — Decoy-path swap + literature verifier (support)

Keep the drug and cancer, swap the real bridge gene for a decoy bridge gene, and run the lexical MOA verifier on Europe PMC literature for each. (No LLM key here — the lexical/MOA-rubric path is used; an LLM judge would strengthen the precision.)

- Verified 30 (drug, cancer) pairs
- **Supported rate**: true = **0.800** vs decoy = **0.167** (separation 0.633)
- Supported|weak rate: true = 0.967 vs decoy = 0.600 (separation 0.367)
- True grade distribution: {'supported': 24, 'weak': 5, 'unknown': 1}
- Decoy grade distribution: {'unknown': 12, 'supported': 5, 'weak': 13}

## Test 4 — True MOA vs plausible hard negatives (specificity)

Graph mechanism-score separation of true MOA pairs from progressively harder, *non-random* negatives.

| Negative class | n | Separation AUROC | neg any-path rate |
|---|---|---|---|
| random | 400 | 0.887 | 6.5% |
| degree_matched | 400 | 0.743 | 50.0% |
| oncology_drug | 400 | 0.870 | 11.2% |
| shared_target | 368 | 0.609 | 79.1% |

## Honest reading

- T1 target-edge ablation: contrast=0.168 (MOA=0.165 vs random=-0.003), faithful=0.872, Wilcoxon p=0.00, AUROC=0.878 -> PASS
- T2 wrong-target substitution: rejection=0.926, true-vs-decoy AUROC=0.923, margin=5.734 -> PASS
- T3 decoy-path verifier: supported true=0.800 vs decoy=0.167 (sep=0.633); supported|weak sep=0.367 -> PASS
- T4 hard negatives: random AUROC=0.887, hardest=shared_target AUROC=0.609 (>0.5 means mechanism signal survives) -> PASS

### Caveats a reviewer would raise

- Tests 1–2 are measured only on held-out (cold-disease, oncology) pairs whose curated DrugMechDB bridge gene is a real `drug_protein` edge — the population where the counterfactual is well-posed, not all predictions. Sample sizes are modest.
- The "score" in tests 1–2 is the mechanism-head logit; drops are in logit space (monotonic, not probability). Removing an edge perturbs both endpoints' embeddings; the random control isolates *this* edge by deleting the same drug's other target edges.
- Test 2 decoys are drug-degree-matched and disease-unassociated, but a decoy could still be a genuine (uncurated) partner; rejection is therefore a conservative lower bound on specificity.
- Test 3 uses the lexical co-mention verifier (no LLM key). Lexical grading over-calls "supported" relative to an LLM judge; the LLM step would sharpen the true-vs-decoy gap. It also depends on Europe PMC abstract coverage (OA full text is sparse).
- Test 4's mechanism score is a hand-designed path score (not learned); the shared-target negative (AUROC ≈ 0.6) is honestly the regime where the graph signal is weakest, because the decoy drug shares the real target.
- All results are hypothesis-generating and not medical advice.
