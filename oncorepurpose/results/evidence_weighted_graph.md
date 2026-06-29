# Evidence-weighted mechanism scoring (full)

_2026-06-29T00:09:37.829397_

## What this shows

The graph's structure-only mechanism score treats a coincidental drug->hub-gene->cancer chain the same as a real, literature-attested mechanism. Here we re-weight each mechanism path by Europe PMC evidence (co-mention strength of both links + recency + bridge-gene specificity, minus a contradiction penalty) and test whether that **load-bearing retrieval layer** sharpens separation -- especially against shared-target hard negatives, where the coincidental direct-target path is exactly the failure mode.

## Separation AUROC: structure-only vs evidence-weighted

Two evidence scores: **ev-chain** is spec-faithful (drug-gene & gene-cancer co-mention x recency x bridge-gene specificity, minus a contradiction penalty). **ev-full** additionally multiplies by indication evidence -- whether the drug is co-mentioned with the cancer at all -- which is the term that can separate shared-target hard negatives (they share the mechanism chain by construction).

| comparison | structure-only | ev-chain (delta) | ev-full (delta) |
|---|---|---|---|
| true vs random | 0.886 | 0.887 (+0.001) | 0.890 (+0.004) |
| true vs shared-target (hard) | 0.618 | 0.639 (+0.021) | 0.742 (+0.124) |

Mean scores -- structure: true 1.991, random 0.166, shared-target 1.645. ev-full: true 30.462, random 1.049, shared-target 10.081.

## Coincidental hub paths demoted by evidence weighting

Each row was the **top path by graph structure** for its (drug, cancer) pair but is NOT the top path after evidence weighting, because its bridge gene is a literature hub (mentioned with everything) and/or the drug is never co-mentioned with the target. These are the coincidences the retrieval layer is meant to catch.

| pair | demoted path (structure-top) | gene total mentions | drug-target hits | evidence-top path |
|---|---|---|---|---|
| Hydrocortisone / precursor T-cell acute lymphoblastic leukemia | `Hydrocortisone --targets--> NR3C1 --interacts--> SET <--associated-- precursor T-cell acute lymphoblastic leukemia` (struct 2.06, ew 5.6332) | 5583671 | 4869 | `Hydrocortisone --targets--> ANXA1 --interacts--> ABL1 <--associated-- precursor T-cell acute lymphoblastic leukemia` (ew 6.6097, drug-target hits 219) |
| Prednisone / acute lymphoblastic/lymphocytic leukemia | `Prednisone --targets--> NR3C1 --interacts--> SET <--associated-- acute lymphoblastic/lymphocytic leukemia` (struct 2.06, ew 5.1408) | 5583671 | 1818 | `Prednisone --targets--> NR3C1 --interacts--> BBC3 <--associated-- acute lymphoblastic/lymphocytic leukemia` (ew 5.3664, drug-target hits 1818) |
| Pemetrexed / malignant mesothelioma | `Pemetrexed --targets--> GART --interacts--> FN1 <--associated-- malignant mesothelioma (disease)` (struct 2.08, ew 4.7268) | 153693 | 110 | `Pemetrexed --targets--> DHFR --interacts--> RXRA <--associated-- malignant mesothelioma (disease)` (ew 4.9550, drug-target hits 671) |
| Desoximetasone / primary cutaneous T-cell lymphoma | `Desoximetasone --targets--> NR3C1 --interacts--> NFKB2 <--associated-- primary cutaneous T-cell lymphoma` (struct 2.05, ew 3.8772) | 51903 | 30 | `Desoximetasone --targets--> NR3C1 --interacts--> STAT5B <--associated-- primary cutaneous T-cell lymphoma` (ew 4.3264, drug-target hits 30) |
| Lonidamine / lung neoplasm | `Lonidamine --targets--> CFTR --interacts--> PPP2R1B <--associated-- lung neoplasm` (struct 2.08, ew 3.9283) | 40006 | 32 | `Lonidamine --targets--> HK1 --interacts--> EGR1 <--associated-- lung neoplasm` (ew 5.6223, drug-target hits 146) |
| Bleomycin / primary cutaneous diffuse large B-cell lymphoma, Leg type | `Bleomycin --targets--> LIG3 --interacts--> CDK1 <--associated-- primary cutaneous diffuse large B-cell lymphoma, Leg type` (struct 2.15, ew 4.0876) | 39821 | 92 | `Bleomycin --targets--> LIG1 --interacts--> CDK1 <--associated-- primary cutaneous diffuse large B-cell lymphoma, Leg type` (ew 4.1121, drug-target hits 97) |

## Coincidental shared-target paths demoted below the true-pair bar

These are shared-target HARD NEGATIVES: a drug that shares a target gene with the true drug for this cancer (so it inherits a strong structural direct-target path) but is **not** an indication. Evidence weighting collapses their score below the true-pair median evidence-weighted score (26.8668), because the drug is barely co-mentioned with the cancer in the literature.

| drug | cancer | coincidental structural path | struct | ew | drug-cancer hits |
|---|---|---|---|---|---|
| Insulin pork | gastrointestinal stromal tumor | `Insulin pork --targets--> RB1 <--associated-- gastrointestinal stromal tumor` | 3.08 | 0.0000 | 0 |
| Daclizumab | undifferentiated carcinoma of the corpus uteri | `Daclizumab --targets--> C1QB <--associated-- undifferentiated carcinoma of the corpus uteri` | 3.07 | 1.6213 | 1 |
| Atorvastatin | fibrolamellar hepatocellular carcinoma | `Atorvastatin --targets--> HMGCR <--associated-- fibrolamellar hepatocellular carcinoma` | 3.05 | 18.2464 | 13 |
| Estradiol benzoate | malignant mesothelioma | `Estradiol benzoate --targets--> ESR1 <--associated-- malignant mesothelioma (disease)` | 3.02 | 19.2268 | 12 |

## Honest reading & caveats

- Sample is 50 pairs/group; AUROCs are indicative, not the full 400-pair headline. Random negatives mostly have no path, so the true-vs-random comparison is already near-saturated and has limited headroom; the **shared-target** comparison is where evidence weighting is supposed to help.
- Co-mention counts are raw Europe PMC `hitCount`s: an exact-phrase pairing can still co-occur for non-mechanistic reasons, so the weight is a soft prior, not proof of mechanism.
- Gene-symbol ambiguity (short symbols, gene/alias collisions) inflates some counts; the specificity term partly compensates but is imperfect.
- The contradiction penalty reuses the lexical scan (see `contradiction_detector.py`) and inherits its noise.
