# OncoEvidence -- Mechanism-Novelty Triage

_OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided Cancer Drug Repurposing._

**Goal (i): rebut the "rediscovered obvious facts" criticism.** Every shortlist candidate is sorted into exactly one novelty bucket using the project's MOA-path extractor and the same DrugMechDB UniProt->HGNC map used for curated agreement. Hypothesis-generating; not medical advice.

- Source: `repurposing_shortlist.json`
- Candidates scored: **30**
- DrugMechDB available: **True** (1071 drugs)
- Promiscuous-hub threshold: drug-degree >= **100**

## Novelty class distribution

| Novelty class | Count | Share |
|---|---|---|
| known_mechanism | 5 | 16.7% |
| known_drug_new_cancer | 8 | 26.7% |
| new_mechanism | 17 | 56.7% |
| unsupported_or_hub_artifact | 0 | 0.0% |

**Non-textbook share** (new_mechanism + known_drug_new_cancer): **25/30 = 83.3%**. Rediscovered-fact share (known_mechanism): 16.7%.

## Example rows

| Drug | Cancer | Novelty | Best MOA path | Why |
|---|---|---|---|---|
| Pseudoephedrine | prostate cancer | known_mechanism | Pseudoephedrine --targets--> ADRB2 <--associated-- prostate cancer | Extracted MOA gene(s) ['ADRB2'] are curated drug->target mechanism for Pseudoephedrine in DrugMechDB -- this is a known mechanism. |
| Tetracosactide | prostate cancer | known_mechanism | Tetracosactide --targets--> MC2R <--associated-- prostate cancer | Extracted MOA gene(s) ['MC2R'] are curated drug->target mechanism for Tetracosactide in DrugMechDB -- this is a known mechanism. |
| Oxytocin | non-small cell lung carcinoma (disease) | known_mechanism | Oxytocin --targets--> OXTR <--associated-- non-small cell lung carcinoma (disease) | Extracted MOA gene(s) ['OXTR'] are curated drug->target mechanism for Oxytocin in DrugMechDB -- this is a known mechanism. |
| Cisplatin | glioblastoma (disease) | known_drug_new_cancer | Cisplatin --targets--> GSTT1 <--associated-- glioblastoma (disease) | Cisplatin is already an indicated oncology drug (e.g. ovarian cancer, hereditary breast ovarian cancer syndrome); a plausible MOA path exists for this new cancer via ['GSTT1'], but the specific drug->gene MOA is not in DrugMechDB. |
| Carboplatin | glioblastoma (disease) | known_drug_new_cancer | Carboplatin --targets--> GSTT1 <--associated-- glioblastoma (disease) | Carboplatin is already an indicated oncology drug (e.g. ovarian cancer, hereditary breast ovarian cancer syndrome); a plausible MOA path exists for this new cancer via ['GSTT1'], but the specific drug->gene MOA is not in DrugMechDB. |
| Tamoxifen | glioblastoma (disease) | known_drug_new_cancer | Tamoxifen --targets--> PRKCA <--associated-- glioblastoma (disease) | Tamoxifen is already an indicated oncology drug (e.g. female breast carcinoma, invasive ductal breast carcinoma); a plausible MOA path exists for this new cancer via ['PRKCA'], but the specific drug->gene MOA is not in DrugMechDB. |
| Probenecid | glioblastoma (disease) | new_mechanism | Probenecid --targets--> SLC22A10 <--associated-- glioblastoma (disease) | Specific MOA path via ['SLC22A10'] exists, but the drug->target MOA is NOT in DrugMechDB (drug covered in DrugMechDB) and Probenecid is not an established oncology drug -- a non-obvious, hypothesis-generating mechanism. |
| Potassium chloride | metastatic melanoma | new_mechanism | Potassium chloride --targets--> SLC12A6 --interacts--> OXSR1 <--associated-- metastatic melanoma | Specific MOA path via ['SLC12A6', 'OXSR1'] exists, but the drug->target MOA is NOT in DrugMechDB (drug absent in DrugMechDB) and Potassium chloride is not an established oncology drug -- a non-obvious, hypothesis-generating mechanism. |
| Isoprenaline | metastatic melanoma | new_mechanism | Isoprenaline --targets--> PIK3R3 --interacts--> CAMK1 <--associated-- metastatic melanoma | Specific MOA path via ['PIK3R3', 'CAMK1'] exists, but the drug->target MOA is NOT in DrugMechDB (drug covered in DrugMechDB) and Isoprenaline is not an established oncology drug -- a non-obvious, hypothesis-generating mechanism. |
