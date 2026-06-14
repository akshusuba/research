# Making the GNN Earn Its Keep: Synthetic Lethality Prediction Where Graph Topology Is the Signal

## Motivation — and how this idea was chosen

This project grew out of a critique of an earlier gut–brain knowledge-graph
project (`research/celiac`): on a small, feature-rich link-prediction task, a
plain MLP could match a heterogeneous GNN, so the graph machinery was hard to
justify. The lesson: **a GNN is only worth its complexity when the prediction
target genuinely depends on graph topology that node features cannot encode.**

To find such a task, we surveyed the award-winning papers of the **Student
Research Institute (SRI) Summer 2025 cohort** (Harvard Undergraduate OpenBio
Laboratory / *STEM Fellowship Journal*). Two observations shaped the idea:

1. **The cohort is cancer- and omics-dominated, and leans on feature-vector
   deep learning.** First place (*CUPNavigator*) trains a DNN on binary somatic
   mutation vectors; second place (*DROID*) is an explicit dual-head **MLP** on
   single-cell RNA-seq; third place (CBD-in-AML) uses classical **network
   pharmacology** (hub-gene degree analysis, not learning). The only GNNs in the
   cohort are off-topic for relational gene biology — a *mechanical* stent
   surrogate and a *brain-imaging* connectome classifier. **No cohort paper
   predicts gene–gene biological relationships with a GNN.**

2. **The KRAS-G12C meta-analysis ("Reassessing the Undruggable") is a direct
   hook.** Synthetic lethality (SL) is *the* computational strategy for drugging
   "undruggable" oncogenes such as KRAS: instead of hitting KRAS, you co-inhibit
   a gene whose loss is lethal *only* in the KRAS-mutant background. The cohort
   named the problem but no one applied the method.

Synthetic lethality is therefore (a) novel relative to the cohort, (b)
thematically central to the cohort's cancer focus, and (c) — most importantly —
a task where **the graph is the entire point**.

## Why a GNN must beat an MLP here (the thesis)

Two genes are **synthetic-lethal** when inhibiting both kills the cell while
inhibiting either alone does not. Mechanistically this arises from **parallel-
pathway redundancy**: A and B sit on redundant branches of the same essential
process, so the cell tolerates losing one branch but not both.

This is, by definition, a property of the *relationship between two genes'
positions in the interaction network*, not of either gene's intrinsic features:

- A **feature-only MLP** sees two gene feature vectors and has no access to
  shared neighbors, parallel pathways, or network distance. It must guess.
- A **GNN** propagates over the interaction graph, so each gene's embedding
  reflects its pathway/module context. It can represent "A and B are on
  redundant branches of the same essential process" — the SL mechanism itself.

We make this falsifiable with a **topology-removal ablation**: if the GNN's
advantage is real, destroying the graph structure must erase it.

## Experimental design

### Part A — Controlled synthetic benchmark (mechanism proof)
A generator builds a gene graph where SL is *defined* by topology: genes form
essential processes, each implemented by several redundant modules; a pair is
SL iff the genes are in **different modules of the same process**. Node features
are deliberately uninformative (Gaussian noise). Hard negatives are same-module
pairs (redundant, not lethal) and cross-process pairs (unrelated). Because the
only SL signal is structural, this isolates the contribution of topology.

### Part B — Real data (SynLethDB via KG4SL)
SynLethDB SL/non-SL pairs plus a leakage-free gene interaction graph built from
KG4SL's `INTERACTS_GiG`, `REGULATES_GrG`, and `COVARIES_GcG` relations (the
`SL_GsG`/`NONSL_GnsG` relations are excluded from message passing). Same models,
splits, and trainer as Part A.

**The degree shortcut.** SynLethDB provides only ~3k explicit non-SL pairs, so
negatives must be sampled. With uniform-random negatives a feature MLP reaches
AUROC ~0.88 by exploiting the fact that SL genes are well-studied hubs (a pure
degree/popularity shortcut) — the very "MLP matches GNN" failure this project
exists to diagnose. We therefore default to **degree-matched negatives**, which
equalize the degree distribution of negatives and positives and force reliance
on topology. Under this honest setting the GNN beats the MLP by a modest but
consistent margin (real SL is noisy and the interaction graph is incomplete),
while the controlled synthetic benchmark provides the clean mechanism proof.

### Splits — the decisive comparison
- **Transductive:** pairs split randomly; all genes seen in training.
- **Inductive (cold-gene):** whole genes held out; test pairs touch unseen
  genes, which are removed from the training message-passing graph and only
  reattached at evaluation. Memorization baselines (node2vec/KGE) cannot embed
  unseen genes and should collapse; an inductive GNN should transfer.

### Models (fair, shared trainer)
- **GNN (ours):** GraphSAGE/GAT/GCN encoder + symmetric MLP decoder.
- **MLP:** identical decoder on node features only — the structure-blind control.
- **node2vec:** random-walk embeddings + the same decoder — structure via
  memorization (transductive only).

### Metrics
AUROC, AUPRC, Hits@{1,3,10}, MRR, over 5 seeds (mean ± std).

## What "success" looks like

| Regime | Expected pattern | Interpretation |
|---|---|---|
| Transductive | GNN ≈ node2vec ≫ MLP | structure carries the SL signal |
| Inductive (cold-gene) | GNN ≫ node2vec ≈ MLP | only the GNN generalizes structurally |
| Topology ablation | GNN(intact) ≫ GNN(rewired) ≈ GNN(empty) ≈ MLP | the win is *topological*, not feature leakage |

A negative result (GNN ≈ MLP) would itself be informative and reported honestly
— but unlike the celiac link-prediction setup, the task is constructed so that a
correctly-built GNN has a mechanism to win.

## Relationship to the SRI 2025 cohort (novelty statement)

- **Distinct task:** no cohort paper addresses synthetic lethality or gene–gene
  relational prediction with a GNN.
- **Distinct method:** cohort computational papers use feature-vector DNN/MLP
  (CUPNavigator, DROID) or classical network analysis (CBD/AML); we use an
  inductive GNN and prove *when* it is necessary.
- **Shared theme:** cancer precision oncology and the KRAS "undruggable"
  problem, connecting the work to the cohort's center of gravity.

## References (method + data)
- Hamilton, Ying, Leskovec. *Inductive Representation Learning on Large Graphs
  (GraphSAGE)*. NeurIPS 2017.
- Wang et al. *SynLethDB 2.0: a web-based knowledge graph database on synthetic
  lethality*. Database 2022.
- Zheng et al. *KG4SL: knowledge graph neural network for synthetic lethality
  prediction in human cancers*. Bioinformatics 2021.
- O'Neil et al. *Synthetic lethality and cancer*. Nat. Rev. Genet. 2017.
