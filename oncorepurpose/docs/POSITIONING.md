# Positioning vs prior work (honest)

OncoEvidence does **not** claim to invent KG-based or LLM-based drug repurposing.
Close prior work exists; we treat it as reference and are explicit about it. The
contribution is an oncology-specific, honestly-evaluated **evidence-triage
pipeline**, not a new field or a "first ever" claim.

## Prior work we build on

- **TxGNN** (Huang, Chandak, ... Zitnik, *Nature Medicine* 2024), a graph
  foundation model for zero-shot repurposing on a PrimeKG-derived medical KG
  (17,080 diseases, 7,957 drugs), with stringent disease-area holdouts (one area
  is "cancerous diseases"), strong zero-shot gains, and a multi-hop **Explainer**.
  This is the incumbent; our inductive cold-disease split mirrors its disease-area
  holdout idea.
- **KGML-xDTD**: drug-treatment prediction plus KG path-based **mechanism of
  action** descriptions. Closest in spirit to our mechanism-path aim.
- **DrugKLM**: biomedical KGs combined with **LLM mechanistic reasoning** for
  therapeutic prioritization. Closest in spirit to our LLM-verification aim.
- **Decagon** (Zitnik, Agrawal, Leskovec): GNNs for relational drug-pair
  (polypharmacy) prediction, the canonical demonstration that graph models beat
  tabular ones when the task is genuinely relational.

## What is (and isn't) new here

Not new: using PrimeKG + a GNN for repurposing; extracting KG paths; using an LLM
over a KG. Those exist.

Our narrower, defensible contributions:

1. **An honest "is the graph even necessary?" audit.** With a properly *tuned*
   XGBoost on the identical features (plus a topology-removal ablation and a
   DistMult memorization control), we show the GNN does **not** beat the tabular
   baseline on the link-ranking task in any regime. Most repurposing papers do not
   foreground this comparison; reporting the negative result is part of the point.

2. **Mechanism-path extraction as the graph's real job.** Because the link task
   doesn't need the graph, we move the graph's value to where a tabular model
   cannot follow: multi-hop `drug → target → (PPI / pathway) → cancer gene →
   cancer` chains, prioritizing mechanism relations over phenotype/symptom
   coincidence. On true indications these recover textbook MOAs (FLT3, RARA,
   TOP2A, TYMS); random pairs mostly yield none.

3. **LLM evidence verification with literature grounding.** The LLM acts as an
   evidence reviewer (supported / weak / contradicted / unknown) over retrieved
   literature, to reject coincidental hub paths (e.g. albumin bridges) and keep
   real mechanisms.

4. **Evaluation, not just a demo.** A falsifiable claim, tested: do mechanism
   paths separate true indications from random drug-cancer pairs? Over 400 vs 400
   oncology pairs the graph mechanism signal separates them at **AUROC 0.879**
   (direct-target rate 34% vs 0.25%). We stress-test this honestly with harder
   negatives: it stays strong against oncology-drug negatives (0.870) but falls to
   **0.609 against shared-target negatives** (a drug whose target is also linked to
   the cancer). So the honest claim is *strong vs random, modest under the hardest
   biologically-similar controls* -- which is what a mature mechanism signal should
   look like. The LLM verifier grades true pairs
   *supported/weak* and random pairs *no-path* (11/50 vs 1/50 *supported*). Crucially,
   the LLM verifier is **precise**: of its *supported* calls that DrugMechDB covers,
   **0.857 are confirmed by the curated MOA** vs only 0.591 for a lexical baseline.
   The extracted paths also agree with **expert-curated DrugMechDB mechanisms at
   0.802** on covered pairs (after a UniProt→HGNC mapping). Many LLM+KG repurposing
   demos report no such evaluation.

5. **Focused oncology scope** with a concrete, ranked, evidence-backed candidate
   shortlist as the deliverable.

## One-paragraph claim (use this)

> Prior work has used biomedical knowledge graphs and GNNs for drug repurposing,
> and newer work is beginning to fuse KGs with LLM reasoning. OncoEvidence
> combines and **evaluates** an oncology-specific, citation-grounded
> evidence-triage pipeline that tests honestly when graph structure is actually
> necessary and checks whether proposed mechanism paths are supported by retrieved
> literature. We combine and evaluate; we do not claim to be first.

## Key references

- Huang, Chandak, ... Zitnik. *A foundation model for clinician-centered drug
  repurposing (TxGNN).* Nature Medicine, 2024.
- *KGML-xDTD: a knowledge graph-based machine learning framework for drug
  treatment prediction and mechanism description.* (arXiv:2212.01384)
- *DrugKLM: knowledge-graph + LLM mechanistic reasoning for therapeutic
  prioritization.* (preprint)
- Zitnik, Agrawal, Leskovec. *Modeling polypharmacy side effects with graph
  convolutional networks (Decagon).* Bioinformatics, 2018.
- Chandak, Huang, Zitnik. *Building a knowledge graph to enable precision medicine
  (PrimeKG).* Scientific Data, 2023.
