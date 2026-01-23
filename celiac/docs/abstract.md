Modeling the Celiac Gut–Brain Axis with a Heterogeneous Graph Neural Network

Integrating Duodenal Transcriptomics, Celiac Microbiome Evidence, Immune Anchors, Neurotransmitter Pathways, Phenotype Ontologies, and a Normative Brain Scaffold

Author: Akshatha Arunkumar

---

## Abstract

I study how celiac disease (CeD) might relate to neurological symptoms by building a celiac‑specific knowledge graph (KG) that brings together duodenal gene expression, CeD‑associated microbiome patterns, immune anchors (HLA‑DQ2/8 and TG6), neurotransmitter/metabolite pathways (e.g., tryptophan → kynurenine/serotonin), standardized neurological phenotypes (Human Phenotype Ontology), and a normative brain connectivity scaffold. I then train a heterogeneous graph neural network (GNN) to (1) predict missing links (gene↔phenotype, microbe↔phenotype) and (2) find short, meaningful multi‑step paths that connect gut microbes to brain‑related phenotypes through known biology. Each reported path is paired with citation‑backed context from curated resources.

What is new: a CeD‑focused, typed KG spanning gut–immune–brain layers; an explainable GNN workflow; and a transparent path‑ranking rulebook that favors stronger evidence, shorter chains, and non‑hub routes.

## **Background & Prior Work**

**What CeD + ML has already addressed (by modality & model):**

- **Intestinal gene expression.** Researchers have used statistics/ML to highlight immune and epithelial signals in duodenal tissue; I use these signals to define **Gene** nodes and features.
- **Microbiome shifts.** Meta‑analyses report **which taxa are enriched or depleted** in CeD across studies; I convert those effect sizes/directions into **Microbe** nodes with simple features.
- **Neurologic involvement.** Reviews summarize **ataxia, peripheral neuropathy, and cognitive changes** in a subset of patients and discuss mechanisms such as **TG6 antibodies** in gluten ataxia—motivating **Phenotype** and **Immune** layers.

**Why these modalities belong together:**

The gut–brain axis is **relational**: microbes make **chemicals** (metabolites/neurotransmitters) that influence **host genes/receptors** and connect to **symptoms** and **brain regions**. Curated **microbiota→metabolite→brain** knowledge graphs already organize these relationships; I adapt that idea specifically to **CeD** so gut and immune signals can be traced to neurological outcomes.

**Why typed GNNs fit (and where they’ve worked):**

Typed GNNs like **GraphSAGE** pass information along **different relation types** and have succeeded in **biomedical link‑prediction** tasks where multi‑hop biology matters. I bring that approach to **CeD gut–brain** modeling, which—so far—has not been assembled as a **CeD‑specific typed KG with an explainable GNN**.

## **Research Questions and Hypotheses**

- **RQ1.** If I encode CeD‑relevant biology as a typed KG, can a relation‑aware GNN **predict** masked **gene↔phenotype** and **microbe↔phenotype** links **better** than strong non‑graph baselines?
    
    **H1.** A heterogeneous GNN will achieve better link‑prediction quality than node2vec/TransE‑style baselines.
    
- **RQ2.** Do the highest‑confidence predictions **trace clear, short paths** that align with known biology (e.g., tryptophan/kynurenine→serotonin; TG6↔ataxia)?
    
    **H2.** Top k‑hop paths will match recognized gut–brain motifs and be supported by curated sources.
    

## **Knowledge‑Graph Formulation (what’s CeD‑specific, why these nodes/edges, how I set directions/weights)**

### A) What goes in (and what is CeD‑specific)

| **Layer (node type)** | **CeD‑specific?** | **Why I include it (one line)** |
| --- | --- | --- |
| **Gene (duodenal)** | **Yes** | Captures host molecular state at the disease site (intestine). |
| **Microbe/Taxon** | **Yes** | Encodes CeD‑enriched/depleted taxa from meta‑analyses. |
| **Immune (HLA‑DQ2/8, TG6)** | **Yes** | Represents CeD immunogenetics and gluten‑ataxia linkage. |
| **Metabolite/Neurotransmitter** | No | Bridges microbes to host biology (e.g., tryptophan → serotonin). |
| **Phenotype (HPO)** | No | Standardized neurologic endpoints and gene↔phenotype edges. |
| **BrainRegion (normative)** | No | Scaffold of brain regions/connectivity for interpretation; not CeD imaging. |

> Summary: Gene, Microbe, Immune are CeD‑specific; Metabolite/NT, Phenotype, BrainRegion are general scaffolds that help connect the dots.
> 

### B) Why these **edge types** and **directions**

- **Flows get arrows:** *Microbe → Metabolite*, *Metabolite → Gene*, *Gene → Phenotype*, *Immune → Phenotype*, *Phenotype → Region*, *Gene → Region*.
- **Symmetric relations stay undirected:** *Gene—Gene* (co‑expression), *Microbe—Microbe* (co‑occurrence), *Region—Region* (functional connectivity).
- **Phenotype→Region anchors** are used **sparingly** (e.g., Ataxia→Cerebellum) and only when literature is strong—so the graph suggests routes, not claims.

### C) How I set **simple, transparent weights** (no heavy math)

To make this explainable to judges, I keep a **three‑level evidence scale** with explicit, pre‑registered criteria:

| **Level** | **Criteria (any of the following)** | **Examples** |
| --- | --- | --- |
| **High** | ≥2 independent replications **OR** odds ratio >3 **OR** FDR q<0.01 | TG6 → Ataxia; HLA‑DQ2 → CeD susceptibility |
| **Medium** | 1 replication **OR** 1.5<OR≤3 **OR** 0.01≤q<0.1 | Most microbe→metabolite edges; metabolite→gene edges |
| **Low** | Single study only **OR** OR≤1.5 **OR** q≥0.1 | Exploratory microbe→phenotype edges |

These thresholds are fixed before model training to prevent post‑hoc adjustments. For co‑expression and connectivity I rescale raw values into **0–1** so they fit the same simple idea. All edges carry **source IDs** and **notes** so I can cite them.

## **Methods of Data Collection (public, de‑identified; first‑person)**

I use **exclusively public, de‑identified datasets** and expert‑curated resources; I do not recruit human participants.

- **Gene expression:** I will obtain **GSE164883** (GEO) and retain **duodenal biopsies**. I will compute log2 fold‑change and FDR‑adjusted q‑values and record accession, version, and download date.
- **Microbiome:** I will extract taxa effect sizes/directions from **peer‑reviewed celiac meta‑analyses**. If I re‑process raw reads (SRA), I will use a single pipeline across studies and record accessions.
- **Phenotypes & gene links:** I will use the **Human Phenotype Ontology (HPO)** and the **Monarch Initiative** exports/APIs for standardized neurologic phenotypes and **gene↔phenotype** edges.
- **Gut–brain routes:** I will incorporate relations from curated **microbiota–gut–brain** KGs (e.g., **MiKG/MMiKG**) to connect microbes to **metabolites/neurotransmitters** and onward to host receptors/genes.
- **Brain scaffold (normative):** I will download **HCP S1200 group‑average functional connectivity** (with **Schaefer‑400** parcellation) for **Region↔Region** edges and map **Gene→Region** weights using the **Allen Human Brain Atlas**.

For **every** source, I keep a **provenance manifest** (accession/DOI, version, license, date) so another student can reproduce my inputs.

## **Methods Overview**

- **Core tasks:**
    
    **(T1)** *Gene ↔ Phenotype* link prediction; **(T2)** *Microbe ↔ Phenotype* link prediction.
    
- **Graph construction & splits:**
    
    Build a **typed, weighted KG**; create **5 leakage‑safe splits** per task (hold out positives by source/study to avoid leakage), and repeat with **3 random seeds**.
    
- **Models compared:**
    
    **Heterogeneous GNN** (GraphSAGE/R‑GCN family) **vs.** strong **non‑graph baselines**: **node2vec + logistic** and **TransE/DistMult + logistic**.
    
- **What I report:**
    
    Standard link‑prediction curves and ranking measures, **calibration** (is confidence reliable?), and **ablations** that remove entire layers (Metabolite, then Immune) to show how much they matter.
    
    *All findings are hypothesis‑generating; I make no diagnostic claims.*

- **Brain scaffold clarification:**
    
    The **Phenotype→Region** edges (e.g., Ataxia→Cerebellum) are **literature‑derived anchors**, not model discoveries. The normative brain connectivity scaffold (HCP S1200) serves as an **interpretive layer** for visualization and contextualization—it is not a source of novel claims. In all reporting, I will clearly separate **discovered paths** (model‑predicted) from **anchored paths** (curated from literature).
    

## **Detailed Experimental Procedure**

**A. Data preparation**

1. **GSE164883:** normalize/transform; compute **log2FC** and **FDR q‑values**; filter low‑count genes (e.g., CPM > 1 in ≥20% of samples); remove unannotated/mitochondrial genes.
2. **Microbiome meta‑analysis:** collect taxa effect sizes; if re‑processing, apply the **same pipeline** across studies; use a standard compositional transform (CLR or arcsine‑square‑root).
3. **Phenotypes/Immune/Pathways:** collect neurologic **HPO** terms and **gene↔phenotype** edges (Monarch); add **HLA‑DQ2/8** and **TG6** nodes with phenotype links; import **microbe→metabolite/NT** and **→gene/phenotype** relations from curated KGs.
4. **Brain scaffold:** create **Region↔Region** adjacency from **HCP S1200**; map **Gene→Region** weights from **AHBA**; add **Phenotype→Region** anchors only when strongly supported (e.g., Ataxia→Cerebellum).

**B. KG construction & splits**

1. Build the typed KG; store as edge lists with relation labels and evidence notes.
2.  Create **5 leakage‑safe splits** per task: mask 15% positives for validation, 15% for test; generate matched negatives; group edges by source/curation to avoid leakage.
3. Repeat each split with **3 seeds**.

**C. Training & baselines**

1. Train **GraphSAGE/R‑GCN** (2–3 layers, 128–256 hidden, dropout 0.2–0.5); decoder = **bilinear/DistMult**; optimizer = **Adam**; early stop by **validation precision–recall**.
2. Train **node2vec + logistic**, **TransE/DistMult + logistic**, and (optional) a simple **flat ML** baseline.
3. Log every run (seed, hyperparameters, metrics); save embeddings and configs.

**D. Evaluation & interpretation**

1. Report **precision–recall and ROC** curves, **ranking** measures, **calibration**, and **abstain** rates (thresholding).
2. **Ablate** layers (remove Metabolite/NT; then Immune) and plot the drop in performance to quantify contribution.
3. Extract **top k‑hop paths** and attach **one‑line citations** for each step; group paths by biological motif (e.g., tryptophan/kynurenine/serotonin; TG6→Ataxia).

**E. Temporal validation (testing genuine predictive power)**

To assess whether the model discovers genuinely novel connections (vs. recapitulating well‑known biology), I will conduct a **temporal hold‑out experiment**:

1. Identify edges that first appeared in literature **after a cutoff year** (e.g., 2021) using publication dates from source databases.
2. Train on **pre‑cutoff edges only**; mask all post‑cutoff edges.
3. Test whether the model predicts post‑cutoff edges at above‑chance rates.
4. Report **temporal AUROC** and **temporal precision@k** separately from standard metrics.

This experiment distinguishes **interpolation** (recapitulating known biology) from **extrapolation** (predicting connections that were later validated).

## **How I Define and Rank “Top” k‑hop Paths (plain‑language rulebook)**

**Goal.** Turn model scores into **clear, citation‑backed mini‑stories**, for example:

*“Bifidobacterium → indole/serotonin → host gene X → Ataxia (Cerebellum).”*

**My ranking rules (no equations):**

1. **Evidence first.** Paths built mainly from **High** and **Medium** evidence links outrank paths that depend on **Low** evidence.
2. **Short and sensible.** I cap at **4 steps** and prefer **shorter chains** when explanations are similar.
3. **No hub cheating.** If a path goes through a node that connects to **too many things** (a “hub”), it **loses points**. This keeps results specific (e.g., **Cerebellum** for ataxia) and avoids generic detours.
4. **Diverse results.** I keep only **a few top paths per phenotype** and remove near‑duplicates so the final set is varied and readable.
5. **Back every path.** I attach **one‑sentence citations** (HPO/Monarch, curated gut–brain KGs, primary papers). If I can’t support a path, I do not promote it.
6. **Sanity checks.** I compare against a **shuffled‑edge** control that preserves node degrees. If top paths still appear after shuffling, I avoid over‑claiming.

## **Ethics, Data Management, and Reproducibility**

- **No human subjects**; all inputs are **public and de‑identified**.
- I will publish a **data manifest** (accession/DOI, version, license, date), a **Git repository** (environment file + exact scripts), and **figure outputs** with clear captions.
- All claims are **research‑only**; I will not offer medical advice.

## **Risks & Mitigations**

| **Risk** | **Impact** | **Mitigation** |
| --- | --- | --- |
| Sparse/heterogeneous data | Lower ceiling on predictive metrics | Use meta‑analytic effect sizes; leakage‑safe splits; report CIs; emphasize interpretability. |
| Overfitting | Inflated validation | 5×3 split‑seed protocol; early stopping; ablations; calibration checks. |
| Path sprawl | Too many trivial paths | Rank by evidence; cap at 4 steps; down‑rank hub routes; require citations. |
| Compute limits | Long training | Cap graph size; 128‑dim hidden; early stopping. |

## **Expected Outcomes**

**Performance expectations:** Given the inherent sparsity of CeD‑specific microbiome and gene‑phenotype data, I anticipate **modest absolute metrics** (AUROC 0.65–0.80 range). The primary contribution is **not** state‑of‑the‑art performance, but rather:

1. A **curated, CeD‑specific multi‑layer KG** with full provenance—a reusable resource for the community.
2. A fair **comparison** showing whether **GNNs** offer measurable benefit over strong non‑graph baselines on **two link‑prediction tasks**. *Negative results (GNN ≈ baselines) are still valuable* and will be reported honestly.
3. A set of **interpretable, literature‑grounded** gut–brain **paths** that **organize existing knowledge** and highlight plausible mechanisms (e.g., tryptophan/kynurenine→serotonin; TG6‑ataxia). These paths are **hypothesis‑generating**, not validated discoveries.
4. **Temporal validation results** showing whether the model can extrapolate to edges discovered after the training cutoff, providing a more rigorous test of predictive power.
5. A complete **reproducible notebook** and figure set (precision–recall/ROC, calibration, ablation bars, t‑SNE of embeddings, example k‑hop path diagram).

*Note: Breakthrough biological findings are unlikely without wet‑lab follow‑up. The value lies in curating and organizing existing knowledge computationally.*

## **Current Progress**

- I built a **Colab prototype** that constructs a demo and a larger KG, trains a **GraphSAGE‑style model** in pure PyTorch, and produces **plots** (precision–recall, calibration), **ablations**, and a **sample multi‑step path** diagram.
- I prepared **Full‑Data** blocks that download **GSE164883** and compute initial gene features (log2FC, q‑values), with stubs for curated microbe→metabolite and phenotype tables.

## **References**

1. Hamilton, W., Ying, Z., & Leskovec, J. (2017). **Inductive Representation Learning on Large Graphs (GraphSAGE).** *NeurIPS.*
2. Giuffrè, M., et al. (2022). **Celiac Disease and Neurological Manifestations: From Gluten to Neuroinflammation.** *International Journal of Molecular Sciences.*
3. Arcila‑Galvis, J. E., et al. (2022). **Comprehensive Mapping of Microbial Biomarkers for Celiac Disease by Meta‑Analysis.** *Frontiers in Microbiology.*
4. Köhler, S., et al. (2021). **The Human Phenotype Ontology in 2021.** *Nucleic Acids Research.*
5. Sun, H., et al. (2023). **MMiKG: A Knowledge‑Graph Platform for Path Mining of Microbiota–Mental Disease Interactions.**