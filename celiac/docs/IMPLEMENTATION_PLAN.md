# NeurIPS/ICLR-Quality Celiac Gut-Brain GNN Implementation Plan

## Goal
Scale from toy graph (225 nodes, 229 edges) to PrimeKG (4M edges) with publication-quality rigor.

## Current Gaps → Solutions

| Gap | Current State | Target |
|-----|---------------|--------|
| No baselines | Only MLP comparison | TransE, DistMult, RotatE, R-GCN, CompGCN, HGT |
| Single seed | One run | 5 seeds with mean ± std |
| Limited metrics | AUROC, AUPRC only | + Hits@K, MRR, precision/recall, F1 |
| No statistical tests | None | Paired t-tests, effect sizes |
| Small graph | 225 nodes | PrimeKG subgraph (~10K-50K nodes) |
| Full-batch training | Can't scale | Mini-batch with LinkNeighborLoader |
| Missing reproducibility | Ad-hoc configs | Structured experiment tracking |

---

## Phase 1: PrimeKG Data Pipeline

### Files to Create
```
celiac/data/primekg/
  __init__.py
  downloader.py           # Download from Harvard Dataverse
  subgraph_extractor.py   # Extract celiac-relevant k-hop subgraph
  pyg_converter.py        # Convert to PyG HeteroData
```

### PrimeKG Details
- **Source**: Harvard Dataverse (https://dataverse.harvard.edu/api/access/datafile/6180620)
- **Size**: 100K+ nodes, 4M+ edges, 29 relation types
- **Format**: CSV with columns: x_id, x_type, x_name, y_id, y_type, y_name, relation

### Subgraph Extraction Strategy
1. **Seed nodes**:
   - Disease: "celiac disease" and related (MONDO ontology)
   - Genes: HLA-DQA1, HLA-DQB1, TGM2, TGM6, IL15, IFNG
   - Phenotypes: Neurological HPO terms (ataxia, neuropathy, etc.)
2. **Expansion**: 2-3 hop neighborhood from seeds
3. **Target size**: 10K-50K nodes, 100K-500K edges

---

## Phase 2: Baseline Models

### Files to Create
```
celiac/baselines/
  __init__.py
  kge_models.py      # TransE, DistMult, RotatE, ComplEx
  gnn_baselines.py   # R-GCN, CompGCN, HGT
  trainer.py         # Unified training interface
```

### Models to Implement

| Model | Type | Implementation |
|-------|------|----------------|
| TransE | KGE | `torch_geometric.nn.kge.TransE` |
| DistMult | KGE | `torch_geometric.nn.kge.DistMult` |
| RotatE | KGE | `torch_geometric.nn.kge.RotatE` |
| ComplEx | KGE | `torch_geometric.nn.kge.ComplEx` |
| R-GCN | GNN | `torch_geometric.nn.RGCNConv` |
| CompGCN | GNN | Custom (Vashishth et al. 2020) |
| HGT | GNN | `torch_geometric.nn.HGTConv` |

---

## Phase 3: Scalable Training

### Key Changes to `celiac/train.py`

```python
from torch_geometric.loader import LinkNeighborLoader

# Mini-batch training for link prediction
loader = LinkNeighborLoader(
    data,
    num_neighbors=[15, 10],  # Neighbors per hop
    edge_label_index=data['gene', 'associated_with', 'phenotype'].edge_label_index,
    edge_label=data['gene', 'associated_with', 'phenotype'].edge_label,
    batch_size=1024,
    neg_sampling_ratio=1.0,
    shuffle=True,
)

# Training loop
for batch in loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    z_dict = model.encode(batch)
    pred = model.decode(z_dict, batch.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
    loss.backward()
    optimizer.step()
```

---

## Phase 4: Evaluation Framework

### Files to Create
```
celiac/evaluation/
  __init__.py
  metrics.py              # Full metrics suite
  experiment_runner.py    # Multi-seed runner
  statistical_tests.py    # Paired t-tests, effect sizes
```

### Metrics Suite

```python
def compute_full_metrics(y_true, y_scores):
    return {
        # Threshold-free
        'auroc': roc_auc_score(y_true, y_scores),
        'auprc': average_precision_score(y_true, y_scores),

        # Ranking metrics
        'hits@1': hits_at_k(y_true, y_scores, k=1),
        'hits@3': hits_at_k(y_true, y_scores, k=3),
        'hits@10': hits_at_k(y_true, y_scores, k=10),
        'mrr': mean_reciprocal_rank(y_true, y_scores),

        # Threshold-based (at optimal F1)
        'precision': precision_at_optimal_threshold(y_true, y_scores),
        'recall': recall_at_optimal_threshold(y_true, y_scores),
        'f1': f1_at_optimal_threshold(y_true, y_scores),
    }
```

### Multi-Seed Protocol

```python
SEEDS = [0, 1, 2, 42, 123]

results = defaultdict(list)
for seed in SEEDS:
    set_seed(seed)
    model = train_model(data, seed=seed)
    metrics = evaluate(model, data)
    for k, v in metrics.items():
        results[k].append(v)

# Report mean ± std
for k, v in results.items():
    print(f"{k}: {np.mean(v):.3f} ± {np.std(v):.3f}")
```

### Statistical Significance

```python
from scipy import stats

def compare_models(model_a_results, model_b_results):
    t_stat, p_value = stats.ttest_rel(model_a_results, model_b_results)
    effect_size = cohen_d(model_a_results, model_b_results)
    return {'t_stat': t_stat, 'p_value': p_value, 'cohens_d': effect_size}
```

---

## Phase 5: Extended Ablation Studies

### Ablations to Run

| Ablation | Purpose |
|----------|---------|
| Node type removal | Importance of each biological entity |
| Edge type removal | Importance of each relation |
| Layer depth (1-4) | Optimal message passing depth |
| Hidden dimension (32, 64, 128, 256) | Capacity requirements |
| Neighbor sampling (5, 10, 15, 20) | Sampling depth impact |
| Negative sampling ratio (1:1, 1:5, 1:10) | Class balance sensitivity |

---

## Phase 6: Interpretability

### Files to Create
```
celiac/interpretability/
  __init__.py
  attention_viz.py      # Attention weight visualization
  path_analysis.py      # Multi-hop path extraction and ranking
  case_studies.py       # Celiac-specific predictions
```

### Path Analysis
Extract and rank multi-hop paths from microbe → phenotype:
1. Find all paths up to k hops
2. Score by: edge evidence weight × path length penalty × node degree penalty
3. Visualize top-k paths for key predictions

---

## Phase 7: Paper Deliverables

### Tables
| Table | Content |
|-------|---------|
| Table 1 | PrimeKG subgraph statistics (nodes/edges by type) |
| Table 2 | Main results: all models × all metrics (mean ± std) |
| Table 3 | Ablation results |
| Table 4 | Statistical significance (p-values) |

### Figures
| Figure | Content |
|--------|---------|
| Fig 1 | Knowledge graph schema |
| Fig 2 | Model architecture diagram |
| Fig 3 | Main results bar chart with error bars |
| Fig 4 | Ablation results |
| Fig 5 | t-SNE of embeddings by node type |
| Fig 6 | Attention/path visualization |
| Fig 7 | Case study: celiac → neurological pathways |

---

## Final Directory Structure

```
celiac/
├── celiac/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── train.py              # Refactor for mini-batch
│   ├── ablations.py          # Extend
│   ├── visualize.py          # Extend
│   ├── data/
│   │   └── primekg/
│   │       ├── __init__.py
│   │       ├── downloader.py
│   │       ├── subgraph_extractor.py
│   │       └── pyg_converter.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── kge_models.py
│   │   ├── gnn_baselines.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── experiment_runner.py
│   │   └── statistical_tests.py
│   └── interpretability/
│       ├── __init__.py
│       ├── attention_viz.py
│       ├── path_analysis.py
│       └── case_studies.py
├── scripts/
│   ├── run_full_experiment.py
│   └── generate_paper_figures.py
├── data/
│   └── primekg/              # Downloaded data
├── models/                   # Saved checkpoints
├── figures/                  # Generated figures
├── results/                  # Experiment results
└── docs/
    ├── neurips/
    ├── iclr/
    └── jei/
```

---

## Execution Order

1. **Data Pipeline**: Download PrimeKG, extract celiac subgraph, convert to PyG
2. **Verify**: Load data, check node/edge counts
3. **Evaluation**: Implement metrics and multi-seed runner
4. **Baselines**: TransE, DistMult first (simplest), then GNN baselines
5. **Scale HetGNN**: Mini-batch training with LinkNeighborLoader
6. **Full Experiments**: All models × 5 seeds on A100
7. **Ablations**: Run full ablation suite
8. **Interpretability**: Generate visualizations and case studies
9. **Paper**: Generate tables/figures, update LaTeX

---

## Compute Requirements

- **GPU**: A100 recommended (40GB+ VRAM for large batches)
- **RAM**: 32GB+ for loading full PrimeKG
- **Storage**: ~5GB for PrimeKG + checkpoints
- **Time estimate**: ~2-4 hours for full experiment suite on A100

---

## Dependencies to Add

```
# requirements.txt additions
torch>=2.0.0
torch_geometric>=2.4.0
torch-sparse>=0.6.17
torch-scatter>=2.1.1
pykeen>=1.10.0          # For additional KGE baselines
scipy>=1.10.0           # For statistical tests
seaborn>=0.12.0         # For publication figures
```
