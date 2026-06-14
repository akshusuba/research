# SynLethGNN — Making the GNN Earn Its Keep on Synthetic Lethality

> A graph neural network is only worth its complexity when the prediction
> target depends on **graph topology** that node features cannot encode.
> Synthetic lethality is exactly such a target — and this repo proves it.

## The idea in one paragraph

Two genes are **synthetic-lethal (SL)** when inhibiting *both* kills a cell but
inhibiting *either alone* does not. Mechanistically, SL comes from **parallel-
pathway redundancy**: the two genes sit on redundant branches of the same
essential process. This is a property of the *relationship between two genes'
positions in the interaction network*, not of either gene's intrinsic features.
So a feature-only MLP has no mechanism to predict it, while a GNN that
propagates over the gene interaction graph does. SynLethGNN builds the data,
models, splits, and ablations to show **where and why** the GNN beats strong
non-graph baselines — most decisively in the **inductive (cold-gene)** setting
and under a **topology-removal ablation**.

This project is a deliberate pivot from a prior gut–brain knowledge-graph
project where a plain MLP could match the GNN. The cohort grounding and full
rationale are in [`docs/PROPOSAL.md`](docs/PROPOSAL.md).

## Headline results (synthetic benchmark, 5 seeds)

| Regime | MLP (features) | node2vec | **GNN (ours)** |
|---|---|---|---|
| Transductive AUROC | ~0.49 (chance) | ~0.97 | **~0.95** |
| **Inductive (cold-gene) AUROC** | ~0.49 (chance) | ~0.50 (chance) | **~0.68** |

**Topology-removal ablation (transductive AUPRC):**
GNN on the intact graph ≈ **0.83**, but on a rewired graph ≈ **0.49** and an
empty graph ≈ **0.50** — collapsing to the structure-blind MLP. The GNN's
advantage is *topological*, not feature leakage.

Interpretation:
- **Transductive:** GNN ≈ node2vec ≫ MLP → structure carries the SL signal.
- **Inductive:** GNN ≫ node2vec ≈ MLP → only the GNN generalizes structurally
  to genes never seen in training (memorization baselines cannot embed them).
- **Ablation:** destroying topology erases the GNN's edge → the win is real.

(Exact numbers from your run are written to `results/` and plotted in
`figures/`.)

## Install

```bash
cd research/synlethgnn
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

This project also runs as-is with the `research/celiac/venv` environment
(torch + torch_geometric already installed).

## Run

**Synthetic demonstration (offline, ~3–15 min on CPU):**

```bash
python scripts/run_synthetic_demo.py          # full: 5 seeds, 300 epochs
python scripts/run_synthetic_demo.py --fast    # quick smoke test: 3 seeds
python scripts/make_figures.py                 # render figures from results/
```

**Real data (SynLethDB via KG4SL, downloads ~95 MB, cached):**

```bash
python scripts/run_real.py                     # download + transductive/inductive
python scripts/run_real.py --no-download        # reuse cached data
```

## How the synthetic benchmark works

Genes are organized into essential **processes**, each implemented by several
redundant **modules**. A pair is labeled SL **iff** the two genes are in
*different modules of the same process* (knocking out both removes all
redundancy → lethal). Hard negatives are *same-module* pairs (still redundant,
not lethal) and *cross-process* pairs (unrelated). **Node features are pure
noise**, so the only SL signal is structural. See `synlethgnn/data/synthetic.py`.

## Real data

`synlethgnn/data/real.py` downloads the KG4SL release of SynLethDB (~24k genes,
~476k interaction edges, ~73k SL pairs) and builds a **leakage-free** gene
graph: SL/non-SL pairs become labels, while the message-passing graph is built
only from gene–gene interaction relations (`INTERACTS_GiG`, `REGULATES_GrG`,
`COVARIES_GcG`). The SL relation never appears in the edges the model sees.

### Real features, strong baselines, and an honest verdict

The real-data experiment is deliberately built to be a **hard, fair** test:

- **Real biological features.** Each gene gets a 128-d functional fingerprint
  from its GO (biological process / molecular function / cellular component)
  and pathway membership (`feature_mode="functional"`, ~68% gene coverage).
  This gives the non-graph baselines genuine signal.
- **Strong baselines.** Besides the structure-blind MLP and node2vec, we
  include **XGBoost** on the same pair features — the baseline the SRI winning
  formula demands.
- **Relation-typed GNN.** An **R-GCN** uses the distinct interaction relations
  (interaction / regulation / co-expression), alongside GraphSAGE.
- **Degree-matched negatives.** SynLethDB has only ~3k explicit non-SL pairs,
  so negatives are sampled with endpoint degrees matched to the positives,
  removing the popularity shortcut a feature model would otherwise exploit
  (`neg_strategy="degree_matched"`, default). Set `feature_mode="noise"` to run
  the adversarial-to-features variant that isolates topology.

**Honest finding (measured, seed 0, full training).** When genes carry rich
functional features, the non-graph baselines are *very strong*:

| Real data (functional features) | XGBoost | MLP | SAGE (GNN) |
|---|---|---|---|
| Transductive AUROC | 0.770 | 0.796 | **0.803** |
| Inductive (cold-gene) AUROC | 0.712 | 0.719 | **0.721** |

Transductively the GNN is a hair ahead of a well-trained MLP (+0.007) and
XGBoost (+0.033); in the decisive **cold-gene** regime the three are
**essentially tied** (~0.72). In other words: **on real, feature-rich SL data,
graph message passing adds little beyond what a strong tabular model extracts
from the features.** This is a deliberately honest, falsifiable result — and it
is consistent with recent critiques showing that SL benchmarks are largely
solvable from features/memorization.

Where the GNN *does* shine, dramatically and provably, is the **controlled
synthetic benchmark** (features carry no SL signal, so topology is the only
route) and the **topology-removal ablation**. The scientific contribution is
therefore **diagnosing *when* the graph is necessary** — not a headline number.

Hardware note: full-batch **R-GCN** on the 24k-gene graph exceeds CPU memory
here; it is implemented (`encoder="rgcn"`, `FastRGCNConv`) and intended to run
on GPU or with neighbor sampling. Numbers above use GraphSAGE as the GNN.
Multi-seed JSON is written to `results/real_comparison_functional.json`.

## Project layout

```
synlethgnn/
├── README.md
├── requirements.txt
├── docs/
│   └── PROPOSAL.md              # cohort-grounded rationale + methods + thesis
├── synlethgnn/
│   ├── config.py                # dataclass configs (paths, seeds, hyperparams)
│   ├── data/
│   │   ├── synthetic.py         # topology-defined SL benchmark generator
│   │   └── real.py              # SynLethDB/KG4SL downloader + graph builder
│   ├── models/
│   │   ├── gnn.py               # GraphSAGE/GAT/GCN/R-GCN encoder + decoders (ours)
│   │   ├── xgboost_baseline.py  # strong tabular baseline (pair features)
│   │   ├── mlp.py               # feature-only MLP (structure-blind control)
│   │   └── node2vec.py          # random-walk embeddings (transductive baseline)
│   ├── splits.py                # transductive + inductive (cold-gene) splits
│   ├── metrics.py               # AUROC, AUPRC, Hits@K, MRR
│   ├── train.py                 # shared trainer (best-of-K restarts)
│   ├── experiment.py            # multi-seed model comparison
│   └── ablation.py              # topology-removal falsification test
├── scripts/
│   ├── run_synthetic_demo.py    # offline thesis demonstration
│   ├── run_real.py              # SynLethDB experiment
│   └── make_figures.py          # figures from results JSON
├── results/                     # JSON outputs + run log
└── figures/                     # generated figures
```

## Why this is a fair comparison

Every model exposes the same `encode` / `decode` API and is trained by the same
loop (identical optimizer, loss, early-stopping criterion, hard negatives, and
metrics). The only thing that differs is the inductive bias. The MLP uses the
*same* decoder as the GNN; node2vec uses the same decoder on random-walk
embeddings. So any gap reflects the value of message passing, nothing else.

## Status & honest caveats

- The synthetic benchmark is a controlled *mechanism proof*: features are noise
  by design, which is the most adversarial case for the MLP. The real-data path
  uses degree+noise features and the genuine SynLethDB labels for a realistic
  test.
- Inductive AUROC (~0.68) is well above chance and far above the baselines, but
  cold-start SL is genuinely hard; this is reported as-is, not inflated.
- A negative result on real data (GNN ≈ MLP) would be reported honestly. Unlike
  feature-rich tasks, the task is *constructed* so a correct GNN has a mechanism
  to win.
