import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
import common as C
from torch_geometric.data import Data

"""PPI-DiseaseGene -- disease-gene prediction on a protein-interaction network.

A protein-protein interaction (PPI) network is generated with a stochastic block
model: genes fall into topological modules (dense within-module interaction,
sparse between). A handful of modules are "disease modules" -- by the
guilt-by-association principle, genes inside them are disease-associated. The
label (disease gene or not) is therefore a property of TOPOLOGICAL module
membership.

Node features are noisy per-gene "expression" measurements only weakly correlated
with disease status -- not enough to classify a gene in isolation. Tabular
baselines see only expression; the GNN additionally propagates over PPI edges,
recovering module membership (and hence disease association) by aggregating the
weak signals of densely-interacting neighbors.
"""

SEED = 0


def main():
    C.set_seed(SEED)
    rng = np.random.default_rng(SEED)
    n_blocks = 25
    block_size = 100
    n = n_blocks * block_size            # 2500 genes
    p_in, p_out = 0.09, 0.0008           # modular PPI connectivity

    sizes = [block_size] * n_blocks
    probs = np.full((n_blocks, n_blocks), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs, seed=SEED)

    block_of = np.empty(n, dtype=int)
    for b, members in enumerate(G.graph["partition"]):
        for v in members:
            block_of[v] = b

    # A few modules are disease modules (guilt-by-association).
    disease_blocks = set(rng.choice(n_blocks, size=6, replace=False).tolist())
    y = np.array([1 if block_of[i] in disease_blocks else 0
                  for i in range(n)], dtype=np.int64)

    # Noisy expression weakly correlated with disease status (insufficient alone).
    yc = (2 * y - 1).astype(np.float64)
    expr = np.stack([
        0.40 * yc + rng.standard_normal(n),
        0.30 * yc + 1.2 * rng.standard_normal(n),
        0.25 * yc + 1.3 * rng.standard_normal(n),
        rng.standard_normal(n),          # pure-noise housekeeping channel
    ], axis=1).astype(np.float32)
    X = (expr - expr.mean(0)) / (expr.std(0) + 1e-6)

    print(f"disease-gene rate = {y.mean():.3f}  (n={n}, edges={G.number_of_edges()})")

    edge_index = torch.tensor(np.array(G.edges).T, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data = Data(x=torch.tensor(X), edge_index=edge_index, y=torch.tensor(y))

    idx = rng.permutation(n)
    ntr, nval = int(0.6 * n), int(0.2 * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:ntr]] = True
    val_mask[idx[ntr:ntr + nval]] = True
    test_mask[idx[ntr + nval:]] = True

    Xtr, ytr = X[train_mask.numpy()], y[train_mask.numpy()]
    Xte, yte = X[test_mask.numpy()], y[test_mask.numpy()]

    xgb = C.run_xgboost(Xtr, ytr, Xte, yte, num_class=2)
    mlp = C.run_mlp(Xtr, ytr, Xte, yte, num_class=2)
    gnn = C.run_gnn_node(data, train_mask, val_mask, test_mask, num_class=2,
                         epochs=200)

    C.log_result("exp03_disease_gene", "node-cls", gnn, mlp, xgb, primary="auc",
                 notes="SBM PPI network; disease genes occupy topological modules; "
                       "GNN recovers guilt-by-association from weak noisy expression.")


if __name__ == "__main__":
    main()
