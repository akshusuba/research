# Positioning vs prior work

## The incumbent: TxGNN (Zitnik et al., Nature Medicine 2024)
TxGNN is a graph foundation model for zero-shot drug repurposing on a
PrimeKG-derived medical knowledge graph (17,080 diseases, 7,957 drugs). It
predicts indications/contraindications, uses stringent **disease-area holdout**
splits (one of nine areas is "cancerous diseases"), beats eight baselines
(incl. RGCN and BioBERT), and ships a multi-hop path **Explainer**.

Our inductive cold-disease split is deliberately the same idea as TxGNN's
disease-area holdout, so results are directly comparable.

## What OncoRepurpose-GNN does differently (under a "win on rigor + deliverable" framing)
We do **not** claim a new architecture. The contribution is threefold:

1. **A rigorous "is the graph necessary?" analysis that TxGNN does not foreground.**
   - A *tuned XGBoost tabular baseline* on the identical shared node features.
   - A *topology-removal ablation* (shuffle / empty graph) proving the GNN's gain
     is topological rather than feature leakage.
   - A *memorization control* (DistMult KGE) shown to collapse on unseen nodes.
   This answers *how much* and *when* the graph earns its keep for oncology
   repurposing -- the central question motivated by an earlier project where a
   plain MLP matched a GNN on a feature-rich link-prediction task.

2. **An agentic evidence-report deliverable.** Top predictions are turned into
   citation-grounded dossiers via RAG over Europe PMC plus the multi-hop KG
   rationale, and triaged by an **LLM-as-judge** plausibility/evidence score.
   This goes beyond a path explorer toward an actionable, vetted shortlist.

3. **Focused oncology scope** with a concrete, ranked candidate list for selected
   cancers.

## Honest framing
The field of KG-based repurposing is mature and crowded; novelty here is in the
*execution, rigor, and deliverable*, not the task. The project is explicitly
designed so the GNN has a real topological mechanism to beat a strong tabular
baseline (multi-hop drug-protein-pathway-disease paths), and the experiments are
built to report honestly if that fails to hold.

## Key references
- Huang, Chandak, ... Zitnik. *A foundation model for clinician-centered drug
  repurposing (TxGNN).* Nature Medicine, 2024.
- Chandak, Huang, Zitnik. *Building a knowledge graph to enable precision medicine
  (PrimeKG).* Scientific Data, 2023.
- Hamilton, Ying, Leskovec. *Inductive Representation Learning on Large Graphs
  (GraphSAGE).* NeurIPS, 2017.
