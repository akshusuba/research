import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
import common as C
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# DDI-Mol: adverse drug substructure detection on molecular graphs (GRAPH-cls)
#
# Each synthetic "molecule" is a graph of ~20-40 atoms. Node features are a
# one-hot atom type + small noise (atom identity is real info, but individually
# insufficient for the label). The adverse / "reactive" label depends ONLY on a
# CONNECTIVITY MOTIF: whether the molecule contains a direct bond between a
# type-0 atom ("reactive group A") and a type-1 atom ("reactive group B") -- the
# reactive substructure responsible for the drug-drug interaction.
#
# Atom types are assigned class-INDEPENDENTLY, so atom-type COUNTS (and node
# feature mean/std) have identical distributions across both classes. The only
# difference is topological adjacency. Therefore a bag-of-counts / aggregated
# feature baseline is at chance, while a GNN that does message passing detects
# the reactive bond and wins.
# ---------------------------------------------------------------------------

N_TYPES = 6
NOISE_DIM = 4
IN_DIM = N_TYPES + NOISE_DIM


def make_molecule(positive: bool):
    n = np.random.randint(20, 41)
    # Class-independent atom-type assignment, guarantee >=2 of type 0 and type 1.
    types = np.random.randint(0, N_TYPES, size=n)
    types[:2] = 0
    types[2:4] = 1
    np.random.shuffle(types)

    G = nx.random_labeled_tree(n, seed=int(np.random.randint(0, 2**31 - 1)))
    n_extra = np.random.randint(n // 4, n // 2 + 1)
    for _ in range(n_extra):
        a, b = np.random.randint(0, n), np.random.randint(0, n)
        if a != b:
            G.add_edge(int(a), int(b))

    # Strip every type0-type1 adjacency so the motif is fully controlled.
    drop = [(u, v) for u, v in G.edges() if {int(types[u]), int(types[v])} == {0, 1}]
    G.remove_edges_from(drop)

    if positive:
        t0 = np.where(types == 0)[0]
        t1 = np.where(types == 1)[0]
        # Plant several reactive bonds so the pooled GNN signal is robust.
        for _ in range(np.random.randint(3, 7)):
            u = int(np.random.choice(t0))
            v = int(np.random.choice(t1))
            G.add_edge(u, v)

    onehot = np.zeros((n, N_TYPES), dtype=np.float32)
    onehot[np.arange(n), types] = 1.0
    noise = (0.25 * np.random.randn(n, NOISE_DIM)).astype(np.float32)
    x = np.concatenate([onehot, noise], axis=1)

    if G.number_of_edges() == 0:
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        e = np.array(list(G.edges()), dtype=np.int64).T
        ei = np.concatenate([e, e[::-1]], axis=1)  # undirected
    return x, ei, int(positive)


def aggregate_features(x):
    """Fair baseline view: per-graph aggregated node features (mean/std) plus
    atom-type COUNTS. No topology is exposed."""
    mean = x.mean(0)
    std = x.std(0)
    counts = x[:, :N_TYPES].sum(0)  # atom-type counts (not topology)
    return np.concatenate([mean, std, counts])


def main():
    C.set_seed(0)
    n_graphs = 1000
    graphs, feats, ys = [], [], []
    for i in range(n_graphs):
        x, ei, y = make_molecule(positive=(i % 2 == 0))
        graphs.append(Data(x=torch.tensor(x),
                           edge_index=torch.tensor(ei),
                           y=torch.tensor([y])))
        feats.append(aggregate_features(x))
        ys.append(y)
    feats = np.stack(feats)
    ys = np.array(ys)

    idx = np.random.permutation(n_graphs)
    n_tr, n_va = int(0.6 * n_graphs), int(0.2 * n_graphs)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    xgb = C.run_xgboost(feats[tr], ys[tr], feats[te], ys[te], num_class=2)
    mlp = C.run_mlp(feats[tr], ys[tr], feats[te], ys[te], num_class=2)

    g = [graphs[i] for i in tr]
    gv = [graphs[i] for i in va]
    gt = [graphs[i] for i in te]
    tl = DataLoader(g, batch_size=32, shuffle=True)
    vl = DataLoader(gv, batch_size=64)
    el = DataLoader(gt, batch_size=64)
    gnn = C.run_gnn_graph(tl, vl, el, in_dim=IN_DIM, num_class=2, epochs=120)

    C.log_result(
        "exp08_ddi", "graph-cls", gnn, mlp, xgb, primary="auc",
        notes=("Synthetic molecular graphs (20-40 atoms). Adverse label = presence "
               "of a reactive type0-type1 bond (connectivity motif). Atom types are "
               "assigned class-independently so atom-type counts and node-feature "
               "mean/std are non-discriminative; only the GNN's message passing "
               "detects the reactive substructure."))


if __name__ == "__main__":
    main()
