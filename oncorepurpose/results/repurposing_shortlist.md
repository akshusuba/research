# OncoEvidence: mechanism-grounded repurposing shortlist

_Candidates are ranked by disease-specific lift and kept only if the knowledge graph yields a mechanism-of-action path (direct target / PPI / shared pathway). Phenotype/symptom coincidences are excluded. Hypothesis-generating; not medical advice._

> **Lead converging-evidence candidate:** **Pimecrolimus → metastatic melanoma** (via mTOR) — the one novel candidate where six independent layers agree (graph MOA path, conformal accept, novel mechanism, PI3K/AKT/mTOR driver context, no literature contradiction, and an independent DepMap CRISPR dependency of MTOR in melanoma cell lines). See `results/converging_evidence.md`.

## glioblastoma (disease)

### Probenecid  (direct-target mechanism | model score 0.795 | specificity lift +0.674)
- MOA path: Probenecid --targets--> SLC22A10 <--associated-- glioblastoma (disease)
- MOA path: Probenecid --targets--> CYP2C9 <--associated-- glioblastoma (disease)
- MOA path: Probenecid --targets--> SLC22A12 --interacts--> MYC <--associated-- glioblastoma (disease)
- MOA path: Probenecid --targets--> SLC16A1 --interacts--> MYC <--associated-- glioblastoma (disease)
- Literature: 5 refs (e.g. Crossing the blood-brain barrier: emerging therapeutic strategies for neurological disease)

### Cisplatin  (direct-target mechanism | model score 0.791 | specificity lift +0.673)
- MOA path: Cisplatin --targets--> GSTT1 <--associated-- glioblastoma (disease)
- MOA path: Cisplatin --targets--> BCHE <--associated-- glioblastoma (disease)
- MOA path: Cisplatin --targets--> CYP2C9 <--associated-- glioblastoma (disease)
- MOA path: Cisplatin --targets--> SOD1 --interacts--> SRRT <--associated-- glioblastoma (disease)
- Literature: 5 refs (e.g. In Vitro Anti-Glioblastoma Activity of a Novel Pt(IV)-Ganoderic Acid A Conjugate.)

### Carboplatin  (direct-target mechanism | model score 0.791 | specificity lift +0.673)
- MOA path: Carboplatin --targets--> GSTT1 <--associated-- glioblastoma (disease)
- MOA path: Carboplatin --targets--> SOD1 --interacts--> SRRT <--associated-- glioblastoma (disease)
- MOA path: Carboplatin --targets--> MT1A --interacts--> SUZ12 <--associated-- glioblastoma (disease)
- MOA path: Carboplatin --targets--> GSTP1 --interacts--> PRKCA <--associated-- glioblastoma (disease)
- Literature: 5 refs (e.g. Metastatic High-Grade Serous Ovarian Carcinoma Presenting as a Temporal Lobe Glioblastoma )

### Tamoxifen  (direct-target mechanism | model score 0.788 | specificity lift +0.672)
- MOA path: Tamoxifen --targets--> PRKCA <--associated-- glioblastoma (disease)
- MOA path: Tamoxifen --targets--> PRKCB <--associated-- glioblastoma (disease)
- MOA path: Tamoxifen --targets--> MAPK8 <--associated-- glioblastoma (disease)
- MOA path: Tamoxifen --targets--> CYP2C9 <--associated-- glioblastoma (disease)
- Literature: 5 refs (e.g. Steroid receptor coactivator 3-deficient regulatory T cells eradicate multiple solid tumor)

### Liothyronine  (direct-target mechanism | model score 0.788 | specificity lift +0.671)
- MOA path: Liothyronine --targets--> PCNA <--associated-- glioblastoma (disease)
- MOA path: Liothyronine --targets--> PCNA --interacts--> POLK <--associated-- glioblastoma (disease)
- MOA path: Liothyronine --targets--> PCNA --interacts--> CDK2 <--associated-- glioblastoma (disease)
- MOA path: Liothyronine --targets--> PCNA --interacts--> FN1 <--associated-- glioblastoma (disease)
- Literature: 5 refs (e.g. Recent advances in drug repurposing for cancer immunomodulation emerging strategies, mecha)

## metastatic melanoma

### Potassium chloride  (interaction-level mechanism | model score 0.878 | specificity lift +0.759)
- MOA path: Potassium chloride --targets--> SLC12A6 --interacts--> OXSR1 <--associated-- metastatic melanoma
- MOA path: Potassium chloride --targets--> SLC12A2 --interacts--> OXSR1 <--associated-- metastatic melanoma
- MOA path: Potassium chloride --targets--> SLC12A1 --interacts--> OXSR1 <--associated-- metastatic melanoma
- Literature: 5 refs (e.g. Enteropathogenic Escherichia coli and Bacterial Overgrowth Co-infection Exacerbating Immun)

### Isoprenaline  (interaction-level mechanism | model score 0.868 | specificity lift +0.758)
- MOA path: Isoprenaline --targets--> PIK3R3 --interacts--> CAMK1 <--associated-- metastatic melanoma
- MOA path: Isoprenaline --targets--> PIK3R1 --interacts--> CAMK1 <--associated-- metastatic melanoma
- MOA path: Isoprenaline --targets--> PIK3R1 --interacts--> STRADB <--associated-- metastatic melanoma
- MOA path: Isoprenaline --targets--> PIK3R1 --interacts--> MAP4K1 <--associated-- metastatic melanoma
- Literature: 5 refs (e.g. β-Adrenergic Signaling Promotes Anti-Tumor Immunity in TP53-mutant Oral Squamous Cell Carc)

### Pseudoephedrine  (interaction-level mechanism | model score 0.870 | specificity lift +0.758)
- MOA path: Pseudoephedrine --targets--> NFATC1 --interacts--> ROCK2 <--associated-- metastatic melanoma
- MOA path: Pseudoephedrine --targets--> ATF4 --interacts--> TAF1 <--associated-- metastatic melanoma
- MOA path: Pseudoephedrine --targets--> ATF3 --interacts--> TAF1 <--associated-- metastatic melanoma
- MOA path: Pseudoephedrine --targets--> FOS --interacts--> TAF1 <--associated-- metastatic melanoma
- Literature: 5 refs (e.g. Transnasal drug delivery to the brain: circumventing barriers for brain tumor patients.)

### Tamoxifen  (direct-target mechanism | model score 0.874 | specificity lift +0.758)
- MOA path: Tamoxifen --targets--> PRKCI <--associated-- metastatic melanoma
- MOA path: Tamoxifen --targets--> PRKCZ --interacts--> RPS6KA2 <--associated-- metastatic melanoma
- MOA path: Tamoxifen --targets--> PRKCZ --interacts--> PRKCI <--associated-- metastatic melanoma
- MOA path: Tamoxifen --targets--> PRKCA --interacts--> GRK7 <--associated-- metastatic melanoma
- Literature: 5 refs (e.g. Cervical Metastasis From Primary Breast Carcinoma: A Case Report and Review of Extragenita)

### Pimecrolimus  (direct-target mechanism | model score 0.875 | specificity lift +0.757)
- MOA path: Pimecrolimus --targets--> MTOR <--associated-- metastatic melanoma
- MOA path: Pimecrolimus --targets--> MTOR --interacts--> STK38 <--associated-- metastatic melanoma
- MOA path: Pimecrolimus --targets--> MTOR --interacts--> OXSR1 <--associated-- metastatic melanoma
- MOA path: Pimecrolimus --targets--> MTOR --interacts--> ULK2 <--associated-- metastatic melanoma
- Literature: 5 refs (e.g. MCSP<sup>+</sup> metastasis founder cells activate immunosuppression early in human melano)

## prostate cancer

### Pamidronic acid  (direct-target mechanism | model score 0.936 | specificity lift +0.830)
- MOA path: Pamidronic acid --targets--> CASP9 <--associated-- prostate cancer
- MOA path: Pamidronic acid --targets--> CASP9 --interacts--> PRKCZ <--associated-- prostate cancer
- MOA path: Pamidronic acid --targets--> CASP9 --interacts--> VCP <--associated-- prostate cancer
- MOA path: Pamidronic acid --targets--> CASP9 --interacts--> EGR1 <--associated-- prostate cancer
- Literature: 5 refs (e.g. Real-world study of medication-related osteonecrosis of the jaw from 2010 to 2023 based on)

### Pseudoephedrine  (direct-target mechanism | model score 0.940 | specificity lift +0.828)
- MOA path: Pseudoephedrine --targets--> ATF3 <--associated-- prostate cancer
- MOA path: Pseudoephedrine --targets--> IL2 <--associated-- prostate cancer
- MOA path: Pseudoephedrine --targets--> ADRB2 <--associated-- prostate cancer
- MOA path: Pseudoephedrine --targets--> ATF3 --interacts--> SDF2L1 <--associated-- prostate cancer
- Literature: 5 refs (e.g. Oncosexology: A Narrative Review on Sexual Health and Quality of Life in Cancer Patients.)

### Tetracosactide  (direct-target mechanism | model score 0.939 | specificity lift +0.828)
- MOA path: Tetracosactide --targets--> MC2R <--associated-- prostate cancer
- MOA path: Tetracosactide --targets--> MC2R --in pathway--> Defective ACTH causes obesity and POMCD <--in pathway-- MC2R <--associated-- prostate cancer
- MOA path: Tetracosactide --targets--> MC2R --in pathway--> Peptide ligand-binding receptors <--in pathway-- GRPR <--associated-- prostate cancer
- MOA path: Tetracosactide --targets--> MC2R --in pathway--> ADORA2B mediated anti-inflammatory cytokines production <--in pathway-- RLN2 <--associated-- prostate cancer
- Literature: 5 refs (e.g. Adrenal insufficiency after long-term high-dose ethinylestradiol use in a transgender woma)

### Zinc chloride  (direct-target mechanism | model score 0.934 | specificity lift +0.827)
- MOA path: Zinc chloride --targets--> SLC39A1 <--associated-- prostate cancer
- MOA path: Zinc chloride --targets--> SELENOP <--associated-- prostate cancer
- MOA path: Zinc chloride --targets--> CLU <--associated-- prostate cancer
- MOA path: Zinc chloride --targets--> MDM2 <--associated-- prostate cancer
- Literature: 5 refs (e.g. Radiolabelled ZnO, Iron Oxide-Based, and Gold Nanoparticles for Cancer Therapy: Synthesis,)

### Nesiritide  (direct-target mechanism | model score 0.941 | specificity lift +0.826)
- MOA path: Nesiritide --targets--> NPR3 <--associated-- prostate cancer
- MOA path: Nesiritide --targets--> NPR2 --interacts--> HSP90AB1 <--associated-- prostate cancer
- MOA path: Nesiritide --targets--> NPR3 --interacts--> NPPA <--associated-- prostate cancer
- MOA path: Nesiritide --targets--> NPR2 --interacts--> MDM2 <--associated-- prostate cancer
- Literature: 5 refs (e.g. Rarely listed essential medicines in 158 national lists.)

## non-small cell lung carcinoma (disease)

### Insulin pork  (direct-target mechanism | model score 0.863 | specificity lift +0.738)
- MOA path: Insulin pork --targets--> RB1 <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Insulin pork --targets--> RB1 --interacts--> E2F5 <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Insulin pork --targets--> RB1 --interacts--> E2F2 <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Insulin pork --targets--> RB1 --interacts--> E2F3 <--associated-- non-small cell lung carcinoma (disease)
- Literature: 5 refs (e.g. Cooked meat-derived extracellular vesicles ssc-miR-1 induces metabolic disorders in the mi)

### Oxytocin  (direct-target mechanism | model score 0.853 | specificity lift +0.737)
- MOA path: Oxytocin --targets--> OXTR <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Oxytocin --targets--> OXT --in pathway--> Vasopressin-like receptors <--in pathway-- OXTR <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Oxytocin --targets--> OXTR --in pathway--> Vasopressin-like receptors <--in pathway-- OXTR <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Oxytocin --targets--> OXT --in pathway--> G alpha (q) signalling events <--in pathway-- OXTR <--associated-- non-small cell lung carcinoma (disease)
- Literature: 5 refs (e.g. Advances in antimicrobial peptides: promising cancer treatments and vaccines.)

### Hydroflumethiazide  (direct-target mechanism | model score 0.854 | specificity lift +0.737)
- MOA path: Hydroflumethiazide --targets--> CA9 <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Hydroflumethiazide --targets--> SLC22A6 --interacts--> ERGIC3 <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Hydroflumethiazide --targets--> CA4 --interacts--> MYC <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Hydroflumethiazide --targets--> ATP1A1 --interacts--> ERBB2 <--associated-- non-small cell lung carcinoma (disease)
- Literature: 5 refs (e.g. The role and mechanism of CHMP4C in poor prognosis and drug sensitivity of lung adenocarci)

### Atosiban  (direct-target mechanism | model score 0.851 | specificity lift +0.737)
- MOA path: Atosiban --targets--> OXTR <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Atosiban --targets--> OXTR --in pathway--> Vasopressin-like receptors <--in pathway-- OXTR <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Atosiban --targets--> AVPR1B --in pathway--> Vasopressin-like receptors <--in pathway-- OXTR <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Atosiban --targets--> AVPR1A --in pathway--> Vasopressin-like receptors <--in pathway-- OXTR <--associated-- non-small cell lung carcinoma (disease)
- Literature: 5 refs (e.g. Immunomodulatory crosstalk between GPCR and hippo signaling in cancer: implications for tu)

### Glycine  (direct-target mechanism | model score 0.855 | specificity lift +0.736)
- MOA path: Glycine --targets--> SHMT2 <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Glycine --targets--> DLD --interacts--> SHMT2 <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Glycine --targets--> SHMT1 --interacts--> SHMT2 <--associated-- non-small cell lung carcinoma (disease)
- MOA path: Glycine --targets--> ALAS1 --interacts--> OXTR <--associated-- non-small cell lung carcinoma (disease)
- Literature: 5 refs (e.g. Prognostic significance of intratumoral neutrophil extracellular traps in pulmonary pleomo)

## colorectal cancer

### Insulin pork  (direct-target mechanism | model score 0.836 | specificity lift +0.711)
- MOA path: Insulin pork --targets--> PCSK2 <--associated-- colorectal cancer
- MOA path: Insulin pork --targets--> CPE <--associated-- colorectal cancer
- MOA path: Insulin pork --targets--> LRP2 <--associated-- colorectal cancer
- MOA path: Insulin pork --targets--> CYP1A2 <--associated-- colorectal cancer
- Literature: 5 refs (e.g. Association of Anti-Inflammatory Dietary Adherence With Biomarkers and Gut Microbiota Rela)

### Insulin human  (direct-target mechanism | model score 0.830 | specificity lift +0.710)
- MOA path: Insulin human --targets--> PCSK2 <--associated-- colorectal cancer
- MOA path: Insulin human --targets--> CPE <--associated-- colorectal cancer
- MOA path: Insulin human --targets--> CYP1A2 <--associated-- colorectal cancer
- MOA path: Insulin human --targets--> RB1 --interacts--> LDB1 <--associated-- colorectal cancer
- Literature: 5 refs (e.g. Fine particulate matter and tobacco product exposure exacerbates metabolic syndrome-relate)

### Tamoxifen  (direct-target mechanism | model score 0.824 | specificity lift +0.708)
- MOA path: Tamoxifen --targets--> PRKCZ <--associated-- colorectal cancer
- MOA path: Tamoxifen --targets--> PRKCE <--associated-- colorectal cancer
- MOA path: Tamoxifen --targets--> PRKCB <--associated-- colorectal cancer
- MOA path: Tamoxifen --targets--> KCNH2 <--associated-- colorectal cancer
- Literature: 5 refs (e.g. ATM-Related Cancer Predisposition)

### Imatinib  (direct-target mechanism | model score 0.826 | specificity lift +0.707)
- MOA path: Imatinib --targets--> ABCA3 <--associated-- colorectal cancer
- MOA path: Imatinib --targets--> SLC22A1 <--associated-- colorectal cancer
- MOA path: Imatinib --targets--> ABCB11 <--associated-- colorectal cancer
- MOA path: Imatinib --targets--> ABCG2 <--associated-- colorectal cancer
- Literature: 5 refs (e.g. Zebularine Boosts Imatinib Efficacy in Cells of Colorectal Cancer via Wnt-Survivin-P-Glyco)

### Metformin  (direct-target mechanism | model score 0.826 | specificity lift +0.707)
- MOA path: Metformin --targets--> ETFDH <--associated-- colorectal cancer
- MOA path: Metformin --targets--> SLC22A1 <--associated-- colorectal cancer
- MOA path: Metformin --targets--> PRKAB1 --interacts--> YBX1 <--associated-- colorectal cancer
- MOA path: Metformin --targets--> PRKAB1 --interacts--> CBR1 <--associated-- colorectal cancer
- Literature: 5 refs (e.g. Interrogating the Mechanistic Link between Neighborhood Deprivation and Colorectal Cancer )

## ovarian carcinoma

### Ivermectin  (direct-target mechanism | model score 0.194 | specificity lift +0.068)
- MOA path: Ivermectin --targets--> GLRA3 <--associated-- ovarian carcinoma
- MOA path: Ivermectin --targets--> ABCC2 --interacts--> FOS <--associated-- ovarian carcinoma
- MOA path: Ivermectin --targets--> GLRA3 --in pathway--> Neurotransmitter receptors and postsynaptic signal transmission <--in pathway-- GLRA3 <--associated-- ovarian carcinoma
- Literature: 5 refs (e.g. Captivating Synergistic, Dose-Dependent Anticancer Effects of Tumor-Regulation Modulators )

### Lapatinib  (direct-target mechanism | model score 0.181 | specificity lift +0.061)
- MOA path: Lapatinib --targets--> ERBB2 <--associated-- ovarian carcinoma
- MOA path: Lapatinib --targets--> ERBB2 --interacts--> KPNB1 <--associated-- ovarian carcinoma
- MOA path: Lapatinib --targets--> EGFR --interacts--> KPNB1 <--associated-- ovarian carcinoma
- MOA path: Lapatinib --targets--> ERBB2 --interacts--> RAF1 <--associated-- ovarian carcinoma
- Literature: 5 refs (e.g. Oncofertility in the Age of HER2 Blockade, Immunotherapy, PARP inhibitors, CDK4/6 inhibito)

### Glycine  (direct-target mechanism | model score 0.177 | specificity lift +0.059)
- MOA path: Glycine --targets--> GLRA3 <--associated-- ovarian carcinoma
- MOA path: Glycine --targets--> DLD --interacts--> KPNB1 <--associated-- ovarian carcinoma
- MOA path: Glycine --targets--> BAAT --interacts--> ERBB2 <--associated-- ovarian carcinoma
- MOA path: Glycine --targets--> DLD --in pathway--> Citric acid cycle (TCA cycle) <--in pathway-- MDH2 <--associated-- ovarian carcinoma
- Literature: 5 refs (e.g. Activity of datopotamab deruxtecan in TROP2-expressing low-grade serous ovarian cancer: a )

### Lindane  (direct-target mechanism | model score 0.176 | specificity lift +0.058)
- MOA path: Lindane --targets--> GLRA3 <--associated-- ovarian carcinoma
- MOA path: Lindane --targets--> ESR1 --interacts--> KPNB1 <--associated-- ovarian carcinoma
- MOA path: Lindane --targets--> PGR --interacts--> FOS <--associated-- ovarian carcinoma
- MOA path: Lindane --targets--> ESR1 --interacts--> EHMT2 <--associated-- ovarian carcinoma
- Literature: 5 refs (e.g. Effect of pesticides on breast cancer tumor.)

### Liothyronine  (interaction-level mechanism | model score 0.174 | specificity lift +0.057)
- MOA path: Liothyronine --targets--> PCNA --interacts--> USP32 <--associated-- ovarian carcinoma
- MOA path: Liothyronine --targets--> PCNA --interacts--> ATAD5 <--associated-- ovarian carcinoma
- MOA path: Liothyronine --targets--> PCNA --interacts--> MDH2 <--associated-- ovarian carcinoma
- MOA path: Liothyronine --targets--> PCNA --in pathway--> TP53 Regulates Transcription of Genes Involved in G2 Cell Cycle Arrest <--in pathway-- TP53 <--associated-- ovarian carcinoma
- Literature: 5 refs (e.g. PPP4C: a potential molecular marker and therapeutic target in thyroid cancer and triple-ne)
