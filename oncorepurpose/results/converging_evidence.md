# OncoEvidence: the single best converging-evidence candidate
> OncoEvidence does not just predict repurposing candidates; it tests whether the proposed mechanism is causally important, biologically specific, and literature-supported. This file surfaces the novel candidate where the most independent evidence layers converge.

**Lead: Pimecrolimus -> metastatic melanoma** (6 independent layers converge; convergence score 6.509).
- **Mechanism (graph):** `Pimecrolimus --targets--> MTOR <--associated-- metastatic melanoma` (direct-target mechanism, score 3.043, specificity lift +0.757).
- **Calibration:** conformal triage = **accept** (confidence 0.73).
- **Novelty:** new_mechanism -- Specific MOA path via ['MTOR'] exists, but the drug->target MOA is NOT in DrugMechDB (drug covered in DrugMechDB) and Pimecrolimus is not an established oncology drug -- a non-obvious, hypothesis-generating mechanism.
- **Oncology driver context:** Pimecrolimus targets MTOR in a PI3K/AKT/mTOR-driven context of metastatic melanoma [driver hit: MTOR].
- **Literature:** 0 supporting vs 0 contradicting sentences (Europe PMC).
- **Functional genomics (independent of graph & text):** bridge gene(s) mean CRISPR dependency **-1.056** in 71 Skin cell lines (more negative = stronger; DepMap).
- Top reference: *MCSP<sup>+</sup> metastasis founder cells activate immunosuppression early in human melanoma metastatic colonization.*

## Top 8 by converging evidence
| Drug | Disease | Layers | Conv. | Mech (lift) | Triage | Novelty | Drivers | Dep (Chronos) |
|---|---|---|---|---|---|---|---|---|
| Pimecrolimus | metastatic melanoma | 6 | 6.509 | 3.043 (+0.757) | accept | new_mechanism | MTOR | -1.056 |
| Hydroflumethiazide | non-small cell lung carcinoma | 6 | 6.504 | 3.112 (+0.737) | accept | new_mechanism | MYC | -0.132 |
| Liothyronine | glioblastoma | 6 | 6.311 | 3.087 (+0.671) | accept | known_drug_new_cancer | CDK6 | -2.629 |
| Nesiritide | prostate cancer | 5 | 5.876 | 3.272 (+0.826) | accept | new_mechanism | MDM2 | 0.08 |
| Zinc chloride | prostate cancer | 5 | 5.818 | 3.215 (+0.827) | accept | new_mechanism | MDM2 | 0.031 |
| Glycine | non-small cell lung carcinoma | 5 | 5.574 | 3.193 (+0.736) | accept | new_mechanism | MDM2 | -0.057 |
| Insulin pork | non-small cell lung carcinoma | 5 | 5.507 | 3.082 (+0.738) | accept | new_mechanism | RB1 | 0.174 |
| Probenecid | glioblastoma | 5 | 5.357 | 3.083 (+0.674) | accept | new_mechanism | MYC | -0.074 |

_Each layer is computed by a separate experiment; agreement across independent layers is the signal. Hypothesis-generating only; not medical advice._
