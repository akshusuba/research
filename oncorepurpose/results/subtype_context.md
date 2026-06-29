# OncoEvidence -- Subtype / Driver-Context Layer

_OncoEvidence: A Counterfactual Evidence-Triage Platform for Mechanism-Guided Cancer Drug Repurposing._

**Goal (ii): make rationales oncology-specific.** A small curated driver list (grouped into mechanistic families) is intersected with each cancer's `disease_protein` neighbours to assign a *driver context*. Candidate rationales are re-expressed in that context, and we report how many candidates have a MOA path that touches a curated driver of that exact cancer. Hypothesis-generating; not medical advice.

## Driver context per cancer

| Cancer | Context families | Driver genes (disease_protein ∩ drivers) |
|---|---|---|
| glioblastoma (disease) | IDH/metabolic, Notch, PI3K/AKT/mTOR, RAS/MAPK, RTK/EGFR, WNT, cell-cycle/RB, kinase-fusion, p53 | ALK, APC, BRAF, CCND1, CDK4, CDK6, CDKN2A, CTNNB1, EGFR, HRAS, IDH1, IDH2, MDM4, MET, MTOR, MYC, NF1, NOTCH1, NOTCH2, NOTCH3, PDGFRA, PIK3CA, PIK3R1, PML, PTEN |
| metastatic melanoma | PI3K/AKT/mTOR, RTK/EGFR, cell-cycle/RB | CDK6, MTOR, PDGFRA |
| prostate cancer | DNA-repair, IDH/metabolic, PI3K/AKT/mTOR, RAS/MAPK, RTK/EGFR, WNT, cell-cycle/RB, hormone, kinase-fusion, p53 | AKT1, APC, AR, ATM, ATR, BRAF, BRCA1, BRCA2, CCND1, CTNNB1, EGFR, ERBB2, ERBB3, ESR1, ESR2, HRAS, IDH1, JAK1, JAK2, KRAS, MDM2, MET, MLH1, MYC, PALB2, PARP1, PIK3CA, PIK3R1, PML, PTEN, RB1, TP53 |
| non-small cell lung carcinoma (disease) | Notch, PI3K/AKT/mTOR, RAS/MAPK, RTK/EGFR, WNT, cell-cycle/RB, p53 | AKT1, AKT2, ALK, APC, BRAF, CDKN2A, E2F1, EGFR, ERBB2, FGFR1, FGFR2, KDR, KIT, KRAS, MAP2K1, MDM2, MET, MTOR, MYC, NOTCH3, NRAS, PIK3CA, PTEN, RAF1, RB1, RET, ROS1, TP53 |
| colorectal cancer | DNA-repair, IDH/metabolic, Notch, PI3K/AKT/mTOR, RAS/MAPK, RTK/EGFR, WNT, cell-cycle/RB, hormone, p53 | AKT1, APC, BRAF, CCND1, CTNNB1, EGFR, ERBB2, ESR2, FGFR3, IDH1, KDR, KRAS, MLH1, MSH2, MTOR, MYC, NF1, NOTCH1, NRAS, PARP1, PIK3CA, PTEN, RET, TP53 |
| ovarian carcinoma | RAS/MAPK, RTK/EGFR, p53 | ERBB2, RAF1, TP53 |

## Candidate driver-context alignment: **14/30 = 46.7%**

| Drug | Cancer | Aligned | Touched driver(s) | Context-aware rationale |
|---|---|---|---|---|
| Probenecid | glioblastoma (disease) | yes | MYC | Probenecid targets SLC22A12 (reaching MYC) in a cell-cycle/RB-driven context of glioblastoma (disease) [driver hit: MYC]. |
| Cisplatin | glioblastoma (disease) | yes | PIK3R1 | Cisplatin targets CYP4A11 (reaching PIK3R1) in a PI3K/AKT/mTOR-driven context of glioblastoma (disease) [driver hit: PIK3R1]. |
| Carboplatin | glioblastoma (disease) | no | - | Carboplatin has a MOA path in glioblastoma (disease) but it does not touch a curated driver of this cancer (['IDH/metabolic', 'Notch', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'kinase-fusion', 'p53']). |
| Tamoxifen | glioblastoma (disease) | no | - | Tamoxifen has a MOA path in glioblastoma (disease) but it does not touch a curated driver of this cancer (['IDH/metabolic', 'Notch', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'kinase-fusion', 'p53']). |
| Liothyronine | glioblastoma (disease) | yes | CDK6 | Liothyronine targets PCNA (reaching CDK6) in a cell-cycle/RB-driven context of glioblastoma (disease) [driver hit: CDK6]. |
| Potassium chloride | metastatic melanoma | no | - | Potassium chloride has a MOA path in metastatic melanoma but it does not touch a curated driver of this cancer (['PI3K/AKT/mTOR', 'RTK/EGFR', 'cell-cycle/RB']). |
| Isoprenaline | metastatic melanoma | no | - | Isoprenaline has a MOA path in metastatic melanoma but it does not touch a curated driver of this cancer (['PI3K/AKT/mTOR', 'RTK/EGFR', 'cell-cycle/RB']). |
| Pseudoephedrine | metastatic melanoma | no | - | Pseudoephedrine has a MOA path in metastatic melanoma but it does not touch a curated driver of this cancer (['PI3K/AKT/mTOR', 'RTK/EGFR', 'cell-cycle/RB']). |
| Tamoxifen | metastatic melanoma | no | - | Tamoxifen has a MOA path in metastatic melanoma but it does not touch a curated driver of this cancer (['PI3K/AKT/mTOR', 'RTK/EGFR', 'cell-cycle/RB']). |
| Pimecrolimus | metastatic melanoma | yes | MTOR | Pimecrolimus targets MTOR in a PI3K/AKT/mTOR-driven context of metastatic melanoma [driver hit: MTOR]. |
| Pamidronic acid | prostate cancer | no | - | Pamidronic acid has a MOA path in prostate cancer but it does not touch a curated driver of this cancer (['DNA-repair', 'IDH/metabolic', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'hormone', 'kinase-fusion', 'p53']). |
| Pseudoephedrine | prostate cancer | no | - | Pseudoephedrine has a MOA path in prostate cancer but it does not touch a curated driver of this cancer (['DNA-repair', 'IDH/metabolic', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'hormone', 'kinase-fusion', 'p53']). |
| Tetracosactide | prostate cancer | no | - | Tetracosactide has a MOA path in prostate cancer but it does not touch a curated driver of this cancer (['DNA-repair', 'IDH/metabolic', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'hormone', 'kinase-fusion', 'p53']). |
| Zinc chloride | prostate cancer | yes | MDM2 | Zinc chloride targets MDM2 in a p53-driven context of prostate cancer [driver hit: MDM2]. |
| Nesiritide | prostate cancer | yes | MDM2 | Nesiritide targets NPR2 (reaching MDM2) in a p53-driven context of prostate cancer [driver hit: MDM2]. |
| Insulin pork | non-small cell lung carcinoma (disease) | yes | RB1 | Insulin pork targets RB1 in a cell-cycle/RB-driven context of non-small cell lung carcinoma (disease) [driver hit: RB1]. |
| Oxytocin | non-small cell lung carcinoma (disease) | no | - | Oxytocin has a MOA path in non-small cell lung carcinoma (disease) but it does not touch a curated driver of this cancer (['Notch', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'p53']). |
| Hydroflumethiazide | non-small cell lung carcinoma (disease) | yes | MYC | Hydroflumethiazide targets CA4 (reaching MYC) in a cell-cycle/RB-driven context of non-small cell lung carcinoma (disease) [driver hit: MYC]. |
| Atosiban | non-small cell lung carcinoma (disease) | no | - | Atosiban has a MOA path in non-small cell lung carcinoma (disease) but it does not touch a curated driver of this cancer (['Notch', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'p53']). |
| Glycine | non-small cell lung carcinoma (disease) | yes | MDM2 | Glycine targets GCAT (reaching MDM2) in a p53-driven context of non-small cell lung carcinoma (disease) [driver hit: MDM2]. |
| Insulin pork | colorectal cancer | no | - | Insulin pork has a MOA path in colorectal cancer but it does not touch a curated driver of this cancer (['DNA-repair', 'IDH/metabolic', 'Notch', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'hormone', 'p53']). |
| Insulin human | colorectal cancer | no | - | Insulin human has a MOA path in colorectal cancer but it does not touch a curated driver of this cancer (['DNA-repair', 'IDH/metabolic', 'Notch', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'hormone', 'p53']). |
| Tamoxifen | colorectal cancer | yes | ESR2 | Tamoxifen targets ESR2 in a hormone-driven context of colorectal cancer [driver hit: ESR2]. |
| Imatinib | colorectal cancer | no | - | Imatinib has a MOA path in colorectal cancer but it does not touch a curated driver of this cancer (['DNA-repair', 'IDH/metabolic', 'Notch', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'hormone', 'p53']). |
| Metformin | colorectal cancer | no | - | Metformin has a MOA path in colorectal cancer but it does not touch a curated driver of this cancer (['DNA-repair', 'IDH/metabolic', 'Notch', 'PI3K/AKT/mTOR', 'RAS/MAPK', 'RTK/EGFR', 'WNT', 'cell-cycle/RB', 'hormone', 'p53']). |
| Ivermectin | ovarian carcinoma | no | - | Ivermectin has a MOA path in ovarian carcinoma but it does not touch a curated driver of this cancer (['RAS/MAPK', 'RTK/EGFR', 'p53']). |
| Lapatinib | ovarian carcinoma | yes | ERBB2 | Lapatinib targets ERBB2 in a RTK/EGFR-driven context of ovarian carcinoma [driver hit: ERBB2]. |
| Glycine | ovarian carcinoma | yes | ERBB2 | Glycine targets BAAT (reaching ERBB2) in a RTK/EGFR-driven context of ovarian carcinoma [driver hit: ERBB2]. |
| Lindane | ovarian carcinoma | yes | ERBB2 | Lindane targets PGR (reaching ERBB2) in a RTK/EGFR-driven context of ovarian carcinoma [driver hit: ERBB2]. |
| Liothyronine | ovarian carcinoma | yes | TP53 | Liothyronine targets PCNA (reaching TP53) in a p53-driven context of ovarian carcinoma [driver hit: TP53]. |
