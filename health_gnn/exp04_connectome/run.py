import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
import common as C
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

"""ConnectomeDx -- neuro-disorder diagnosis from brain functional connectivity.

Each subject is a brain graph over a fixed set of ROIs (40 regions partitioned
into 4 functional modules). The two classes differ ONLY in CONNECTIVITY
STRUCTURE:
  * healthy  -> modular connectivity (dense within-module, sparse between),
  * disorder -> degraded modularity (edges rewired ~Erdos-Renyi at the SAME
                overall edge density), so the modules dissolve.

Node features are per-ROI signal summaries tied to each ROI's module identity.
Crucially, ROI->module assignment is FIXED across all subjects and features do
NOT depend on the class, so the per-graph marginal feature distribution (mean /
std / quantiles) is statistically identical for both classes. Tabular baselines
receive only those topology-free aggregated stats and are therefore at chance.

The module feature vectors are designed to SUM TO ZERO across modules: on a
modular graph each ROI keeps its large, "pure" module signal (which survives the
GNN's ReLU), whereas on a degraded graph each ROI aggregates a near-zero mixture
of opposing module signals (clipped away by ReLU). The graph-level readout
magnitude thus encodes modularity -- visible to the GNN's message passing but
invisible to feature-marginal baselines.
"""

SEED = 0
N_ROI = 40
N_MOD = 4
PER_MOD = N_ROI // N_MOD

# Fixed ROI -> module map (same brain atlas for everyone).
MODULE_OF = np.repeat(np.arange(N_MOD), PER_MOD)

# Module signal templates that sum to zero across modules (balanced marginals).
MODULE_VEC = np.array([
    [2.0, 2.0, 0.0, 0.0],
    [-2.0, -2.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 2.0],
    [0.0, 0.0, -2.0, -2.0],
], dtype=np.float64)
FDIM = MODULE_VEC.shape[1]


def make_node_features(rng):
    """Per-ROI signal summary = its module template + noise. Independent of class
    and identically distributed across nodes for every subject."""
    x = MODULE_VEC[MODULE_OF] + 0.6 * rng.standard_normal((N_ROI, FDIM))
    return x.astype(np.float32)


def modular_adj(rng, p_in=0.45, p_out=0.04):
    A = np.zeros((N_ROI, N_ROI), dtype=bool)
    for i in range(N_ROI):
        for j in range(i + 1, N_ROI):
            p = p_in if MODULE_OF[i] == MODULE_OF[j] else p_out
            if rng.random() < p:
                A[i, j] = A[j, i] = True
    return A


def random_adj(rng, p):
    A = np.zeros((N_ROI, N_ROI), dtype=bool)
    for i in range(N_ROI):
        for j in range(i + 1, N_ROI):
            if rng.random() < p:
                A[i, j] = A[j, i] = True
    return A


def adj_to_edge_index(A):
    src, dst = np.nonzero(A)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def build_dataset(n_graphs, rng):
    # Density-match the degraded class to the modular class.
    n_within = N_MOD * (PER_MOD * (PER_MOD - 1) // 2)
    n_between = N_ROI * (N_ROI - 1) // 2 - n_within
    exp_edges = 0.45 * n_within + 0.04 * n_between
    p_er = exp_edges / (N_ROI * (N_ROI - 1) / 2)

    graphs = []
    for g in range(n_graphs):
        cls = g % 2
        if cls == 0:
            A = modular_adj(rng)
        else:
            A = random_adj(rng, p_er)
        x = make_node_features(rng)
        edge_index = adj_to_edge_index(A)
        if edge_index.numel() == 0:  # avoid empty-graph degenerate case
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data = Data(x=torch.tensor(x), edge_index=edge_index,
                    y=torch.tensor([cls], dtype=torch.long))
        graphs.append(data)
    return graphs, p_er


def graph_feature_stats(data):
    """Topology-FREE per-graph features: mean/std/quantiles of node features."""
    X = data.x.numpy()
    qs = np.quantile(X, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0).reshape(-1)
    return np.concatenate([X.mean(0), X.std(0), qs]).astype(np.float32)


def main():
    C.set_seed(SEED)
    rng = np.random.default_rng(SEED)
    n_graphs = 800

    graphs, p_er = build_dataset(n_graphs, rng)
    print(f"built {n_graphs} brain graphs (40 ROIs); "
          f"modular p_in/p_out=0.45/0.04 vs degraded ER p={p_er:.3f}")

    idx = rng.permutation(n_graphs)
    ntr, nval = int(0.6 * n_graphs), int(0.2 * n_graphs)
    tr_idx, val_idx, te_idx = idx[:ntr], idx[ntr:ntr + nval], idx[ntr + nval:]

    # Tabular baseline features: aggregated node-feature stats only (no topology).
    Xstat = np.stack([graph_feature_stats(g) for g in graphs])
    y = np.array([int(g.y) for g in graphs])
    Xtr, ytr = Xstat[tr_idx], y[tr_idx]
    Xte, yte = Xstat[te_idx], y[te_idx]

    xgb = C.run_xgboost(Xtr, ytr, Xte, yte, num_class=2)
    mlp = C.run_mlp(Xtr, ytr, Xte, yte, num_class=2)

    train_loader = DataLoader([graphs[i] for i in tr_idx], batch_size=32, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=64)
    test_loader = DataLoader([graphs[i] for i in te_idx], batch_size=64)
    gnn = C.run_gnn_graph(train_loader, val_loader, test_loader,
                          in_dim=FDIM, num_class=2, epochs=120)

    C.log_result("exp04_connectome", "graph-cls", gnn, mlp, xgb, primary="auc",
                 notes="Modular vs degraded brain connectivity at matched edge "
                       "density; node-feature marginals identical across classes, "
                       "so only the edge-reading GNN detects loss of modularity.")


if __name__ == "__main__":
    main()
