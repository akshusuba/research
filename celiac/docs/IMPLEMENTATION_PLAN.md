# Hybrid Publication Plan: Celiac Gut-Brain Knowledge Graph with Heterogeneous GNNs

## Vision

A single, rigorous paper that works for **both** clinical journals (JEI, Gut) **and** top ML venues (NeurIPS, ICML, ICLR).

**Title**: *Heterogeneous Graph Neural Networks for Modeling the Celiac Gut-Brain Axis: From Curated Knowledge Graphs to Biomedical Discovery*

---

## Dual-Track Strategy

| Aspect | Clinical Angle (JEI) | ML Angle (NeurIPS/ICLR) |
|--------|---------------------|-------------------------|
| **Contribution** | First CeD-specific gut-brain KG | Domain-driven heterogeneous KG benchmark |
| **Primary dataset** | Curated CeD graph (interpretable) | + PrimeKG subgraph (scalable) |
| **Emphasis** | Biological pathway validation | Model comparison + ablations |
| **Interpretability** | Case studies (TG6→ataxia, tryptophan→serotonin) | Attention visualization, path ranking |
| **Metrics focus** | AUROC, AUPRC (clinical relevance) | + Hits@K, MRR (KG benchmarks) |

---

## Current State → Target

| Dimension | Current | Target | Gap |
|-----------|---------|--------|-----|
| Baselines | None | 7 models | TransE, DistMult, RotatE, node2vec, R-GCN, CompGCN, HGT |
| Seeds | 1 | 5 | Multi-seed with mean ± std |
| Metrics | AUROC, AUPRC | 9 metrics | + Hits@1/3/10, MRR, P/R/F1 |
| Statistical tests | None | Full | Paired t-tests, Cohen's d |
| Datasets | Curated (225 nodes) | 2 datasets | + PrimeKG subgraph (10K-50K nodes) |
| Training | Full-batch | Mini-batch | LinkNeighborLoader for scalability |
| Interpretability | t-SNE only | Full suite | Attention viz, path analysis, case studies |
| Reproducibility | Ad-hoc | Colab notebook | One-click reproducibility |

---

## Phase 1: Evaluation Framework

**Priority**: Build rigorous evaluation before adding models.

### 1.1 Metrics Suite (`celiac/evaluation/metrics.py`)

```python
def compute_all_metrics(y_true, y_scores, y_pred=None):
    """
    Full metrics suite for link prediction.

    Returns:
        dict with: auroc, auprc, hits@1, hits@3, hits@10, mrr, precision, recall, f1
    """
```

### 1.2 Multi-Seed Runner (`celiac/evaluation/experiment_runner.py`)

```python
SEEDS = [0, 1, 2, 42, 123]

def run_multi_seed_experiment(model_class, data, config, seeds=SEEDS):
    """Run experiment across multiple seeds, return aggregated results."""
    results = {metric: [] for metric in METRIC_NAMES}
    for seed in seeds:
        set_all_seeds(seed)
        model = train_model(model_class, data, config, seed)
        metrics = evaluate_model(model, data)
        for k, v in metrics.items():
            results[k].append(v)

    return {
        k: {'mean': np.mean(v), 'std': np.std(v), 'values': v}
        for k, v in results.items()
    }
```

### 1.3 Statistical Tests (`celiac/evaluation/statistical_tests.py`)

```python
def compare_models(results_a, results_b, metric='auroc'):
    """Paired t-test with effect size."""
    t_stat, p_value = stats.ttest_rel(results_a[metric], results_b[metric])
    cohens_d = (np.mean(results_a[metric]) - np.mean(results_b[metric])) / pooled_std
    return {'t_stat': t_stat, 'p_value': p_value, 'cohens_d': cohens_d}
```

---

## Phase 2: Baseline Models

### 2.1 Knowledge Graph Embedding Models (`celiac/baselines/kge_models.py`)

| Model | Type | Key Idea | PyG Support |
|-------|------|----------|-------------|
| **TransE** | Translation | h + r ≈ t | `torch_geometric.nn.kge.TransE` |
| **DistMult** | Bilinear | h ⊙ r ⊙ t | `torch_geometric.nn.kge.DistMult` |
| **RotatE** | Rotation | h ∘ r ≈ t (complex) | `torch_geometric.nn.kge.RotatE` |
| **ComplEx** | Complex | Re(h ⊙ r ⊙ t̄) | `torch_geometric.nn.kge.ComplEx` |

### 2.2 Non-Graph Baseline (`celiac/baselines/node2vec_baseline.py`)

| Model | Type | Key Idea |
|-------|------|----------|
| **node2vec** | Random walk | Skip-gram on graph walks |

### 2.3 GNN Baselines (`celiac/baselines/gnn_baselines.py`)

| Model | Type | Key Idea | Implementation |
|-------|------|----------|----------------|
| **R-GCN** | Relational GNN | Relation-specific weight matrices | `RGCNConv` |
| **CompGCN** | Composition GNN | Composition operators on relations | Custom |
| **HGT** | Heterogeneous Transformer | Type-specific attention | `HGTConv` |

### 2.4 Unified Interface (`celiac/baselines/trainer.py`)

```python
class BaselineTrainer:
    """Unified training interface for all baselines."""

    def __init__(self, model_name: str, data: HeteroData, config: dict):
        self.model = self._create_model(model_name)

    def train(self, epochs: int) -> dict:
        """Train and return metrics."""

    def evaluate(self) -> dict:
        """Evaluate on test set."""
```

---

## Phase 3: Datasets

### 3.1 Curated CeD Graph (Primary)

**Current**: 225 nodes, 229 edges
- **Strength**: Every node/edge is biologically relevant
- **Use**: Primary results, interpretability, case studies

**Enhancements**:
- Expand via Monarch Initiative API (more gene-phenotype edges)
- Add confidence scores to edges
- Target: ~500-1000 nodes, ~2000-5000 edges

### 3.2 PrimeKG Subgraph (Scalability)

**Source**: Harvard Dataverse
- Full graph: 100K+ nodes, 4M+ edges

**Subgraph Extraction** (`celiac/data/primekg/`):

```python
def extract_ced_subgraph(primekg_path: str, hops: int = 2) -> HeteroData:
    """
    Extract celiac-relevant subgraph from PrimeKG.

    Seed nodes:
    - Disease: "celiac disease" (MONDO:0005130)
    - Genes: HLA-DQA1, HLA-DQB1, TGM2, TGM6, IL15, IFNG, TPH1, TPH2
    - Phenotypes: Neurological HPO terms

    Returns:
        HeteroData with ~10K-50K nodes
    """
```

---

## Phase 4: Extended Ablations

### 4.1 Ablation Matrix

| Ablation Type | Variations | Purpose |
|---------------|------------|---------|
| **Node types** | Remove gene / microbe / metabolite / phenotype | Which entities matter most? |
| **Edge types** | Remove each relation type | Which relations matter most? |
| **GNN depth** | 1, 2, 3, 4 layers | Optimal message passing hops |
| **Hidden dim** | 32, 64, 128, 256 | Capacity requirements |
| **Attention heads** | 1, 2, 4, 8 | For HGT/GAT variants |
| **Negative sampling** | 1:1, 1:3, 1:5 | Class balance sensitivity |

### 4.2 Ablation Runner

```python
def run_ablation_suite(base_config: dict, data: HeteroData) -> pd.DataFrame:
    """Run all ablations and return results DataFrame."""
    ablations = [
        ('full', {}),
        ('no_microbe', {'remove_node_types': ['microbe']}),
        ('no_metabolite', {'remove_node_types': ['metabolite']}),
        ('layers_1', {'num_layers': 1}),
        ('layers_3', {'num_layers': 3}),
        ('hidden_32', {'hidden_channels': 32}),
        ('hidden_256', {'hidden_channels': 256}),
        # ... more ablations
    ]
    return run_experiments(ablations, base_config, data)
```

---

## Phase 5: Interpretability

### 5.1 Attention Visualization (`celiac/interpretability/attention_viz.py`)

```python
def visualize_attention_weights(model, data, edge_type, top_k=20):
    """Visualize attention weights for top-k edges."""
```

### 5.2 Path Analysis (`celiac/interpretability/path_analysis.py`)

```python
def extract_top_paths(
    data: HeteroData,
    source_type: str,
    target_type: str,
    model: nn.Module,
    max_hops: int = 3,
    top_k: int = 10
) -> List[Path]:
    """
    Extract and rank multi-hop paths.

    Scoring: edge_score × path_length_penalty × node_degree_penalty
    """
```

### 5.3 Case Studies (`celiac/interpretability/case_studies.py`)

**Key biological validations**:

| Prediction | Expected Path | Biological Support |
|------------|---------------|-------------------|
| TGM6 → Ataxia | TGM6 → anti-TG6 antibodies → cerebellar damage | Hadjivassiliou 2008 |
| Tryptophan → Depression | Trp → 5-HT depletion → mood disorders | Russo 2012 |
| Prevotella → Cognition | Prevotella ↓ → SCFA ↓ → neuroinflammation | Caminero 2019 |

---

## Phase 6: Paper Deliverables

### 6.1 Tables

| Table | Content | Venue Focus |
|-------|---------|-------------|
| **T1** | Dataset statistics (nodes/edges by type) | Both |
| **T2** | Main results: all models × all metrics (mean ± std) | ML |
| **T3** | Curated graph results with biological interpretation | Clinical |
| **T4** | Ablation results | ML |
| **T5** | Statistical significance (p-values, effect sizes) | ML |
| **T6** | Top predicted links with literature support | Clinical |

### 6.2 Figures

| Figure | Content | Venue Focus |
|--------|---------|-------------|
| **F1** | Knowledge graph schema (node/edge types) | Both |
| **F2** | Model architecture (HeteroGNN + decoders) | ML |
| **F3** | Main results bar chart with error bars | Both |
| **F4** | Ablation heatmap | ML |
| **F5** | t-SNE embeddings colored by node type | Both |
| **F6** | Attention/path visualization | Both |
| **F7** | Case study: microbe → metabolite → gene → phenotype paths | Clinical |

---

## Phase 7: Google Colab Notebook

### 7.1 Notebook Structure (`notebooks/celiac_gut_brain_gnn.ipynb`)

```
1. Setup & Installation
   - Mount Drive / clone repo
   - Install dependencies
   - Check GPU availability

2. Data Loading
   - Load curated CeD graph
   - (Optional) Download PrimeKG subgraph
   - Visualize graph statistics

3. Model Training
   - Train HeteroGNN (our model)
   - Train baselines (TransE, R-GCN, etc.)
   - Multi-seed experiments

4. Evaluation
   - Compute all metrics
   - Statistical comparisons
   - Generate results tables

5. Ablation Studies
   - Run ablation suite
   - Visualize ablation results

6. Interpretability
   - t-SNE visualization
   - Path analysis
   - Case studies

7. Export Results
   - Save figures
   - Export tables to LaTeX
```

### 7.2 Colab Compatibility

```python
# Auto-detect environment
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install torch-geometric torch-sparse torch-scatter
    %cd /content/drive/MyDrive/celiac
else:
    # Local/GCP VM execution
    pass
```

---

## Final Directory Structure

```
celiac/
├── celiac/
│   ├── __init__.py
│   ├── config.py                    # Configuration constants
│   ├── models.py                    # HeteroGNN (our model)
│   ├── train.py                     # Training pipeline
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── kge_models.py            # TransE, DistMult, RotatE, ComplEx
│   │   ├── node2vec_baseline.py     # node2vec
│   │   ├── gnn_baselines.py         # R-GCN, CompGCN, HGT
│   │   └── trainer.py               # Unified training interface
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Full metrics suite
│   │   ├── experiment_runner.py     # Multi-seed runner
│   │   └── statistical_tests.py     # Paired t-tests, effect sizes
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── primekg/
│   │       ├── __init__.py
│   │       ├── downloader.py        # Download from Harvard Dataverse
│   │       ├── subgraph_extractor.py # Extract CeD-relevant subgraph
│   │       └── pyg_converter.py     # Convert to PyG HeteroData
│   │
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── attention_viz.py         # Attention weight visualization
│   │   ├── path_analysis.py         # Multi-hop path extraction
│   │   └── case_studies.py          # Biological validation
│   │
│   └── ablations.py                 # Extended ablation framework
│
├── notebooks/
│   └── celiac_gut_brain_gnn.ipynb   # Google Colab notebook
│
├── scripts/
│   ├── run_full_experiment.py       # CLI for full experiment suite
│   └── generate_paper_figures.py    # Generate all paper figures
│
├── data/
│   ├── processed/pyg/               # Curated CeD graph
│   └── primekg/                     # Downloaded PrimeKG data
│
├── models/                          # Saved checkpoints
├── figures/                         # Generated figures
├── results/                         # Experiment results (JSON/CSV)
│
└── docs/
    ├── IMPLEMENTATION_PLAN.md       # This file
    ├── neurips/main.tex
    ├── iclr/main.tex
    └── jei/main.tex
```

---

## Execution Order

### Week 1: Foundation
1. ✅ **Evaluation framework**: metrics.py, experiment_runner.py, statistical_tests.py
2. ✅ **Refactor existing code**: Ensure clean interfaces

### Week 2: Baselines
3. ✅ **KGE baselines**: TransE, DistMult, RotatE (use PyG implementations)
4. ✅ **node2vec baseline**: Random walk embeddings
5. ✅ **GNN baselines**: R-GCN, CompGCN, HGT

### Week 3: Experiments
6. ✅ **Multi-seed experiments**: All models × 5 seeds on curated graph
7. ✅ **PrimeKG pipeline**: Download, extract subgraph, convert
8. ✅ **PrimeKG experiments**: Scale test on larger graph

### Week 4: Analysis & Paper
9. ✅ **Ablations**: Full ablation suite
10. ✅ **Interpretability**: Attention viz, path analysis, case studies
11. ✅ **Colab notebook**: One-click reproducibility
12. ✅ **Paper**: Generate tables/figures, finalize LaTeX

---

## Compute Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | T4 (16GB) | A100 (40GB) |
| RAM | 16GB | 32GB |
| Storage | 10GB | 20GB |
| Time (full suite) | ~4 hours | ~1 hour |

---

## Dependencies

```txt
# Core
torch>=2.0.0
torch_geometric>=2.4.0
torch-sparse>=0.6.17
torch-scatter>=2.1.1

# Baselines
node2vec>=0.4.0

# Evaluation
scipy>=1.10.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0

# Notebook
ipywidgets>=8.0.0
```

---

## Success Criteria

### For JEI (Clinical Journal)
- [ ] Clear biological motivation and novelty
- [ ] Interpretable case studies with literature support
- [ ] Curated graph with transparent methodology
- [ ] Clinical implications in discussion

### For NeurIPS/ICML/ICLR (ML Venues)
- [ ] 7+ baselines with fair comparison
- [ ] 5 seeds with mean ± std
- [ ] Statistical significance tests
- [ ] Scalability demonstrated on PrimeKG
- [ ] Comprehensive ablations
- [ ] Reproducible via Colab notebook

### Both Venues
- [ ] Clean, well-documented code
- [ ] Publication-quality figures
- [ ] Complete experimental protocol
