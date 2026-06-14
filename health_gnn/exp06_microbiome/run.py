import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import common as C

"""
MicrobiomeNet: disease vs. healthy classification from microbial co-occurrence
networks (graph classification).

Story
-----
Each sample is a microbial co-occurrence network: nodes are taxa (~40-50),
edges are significant co-occurrence relationships. The healthy microbiome is
organized into functional MODULES -- taxa co-occur within their module
(assortative wiring). In disease, this modular organization breaks down and the
network reorganizes around a few promiscuous hub taxa that co-occur across
modules (disassortative / hub-dominated wiring).

Crucially, per-taxon abundance marginals are IDENTICAL across the two classes:
each taxon's module membership and abundance profile are drawn the same way
regardless of disease status -- only the co-occurrence EDGES differ. Therefore
baselines that see only aggregated node-feature statistics (mean/std/quantiles
of the abundance matrix) are blind to the difference and sit near chance. The
GNN passes messages over the co-occurrence edges and reads the module-vs-hub
structure directly -> wins.
"""

N_MODULES = 3
FEAT_NOISE = 0.8


def make_graph(rng, disease: bool):
    n = int(rng.integers(40, 51))
    module = rng.integers(0, N_MODULES, size=n)

    # node features: abundance profile tied to module identity + noise.
    # Generated IDENTICALLY for both classes -> identical aggregate marginals.
    centroids = np.eye(N_MODULES, dtype=np.float32) * 2.0
    x = centroids[module] + rng.normal(0, FEAT_NOISE, size=(n, N_MODULES)).astype(np.float32)
    # two pure-noise abundance summaries (still class-agnostic)
    extra = rng.normal(0, 1, size=(n, 2)).astype(np.float32)
    x = np.concatenate([x, extra], axis=1)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    if not disease:
        # HEALTHY: modular / assortative co-occurrence.
        p_in, p_out = 0.45, 0.02
        for a in range(n):
            for b in range(a + 1, n):
                p = p_in if module[a] == module[b] else p_out
                if rng.random() < p:
                    G.add_edge(a, b)
    else:
        # DISEASE: hub-reorganized / module structure dissolved.
        # A few hub taxa connect broadly across modules; remaining edges are
        # rewired at random (disassortative), giving the SAME overall edge
        # density range but a very different topology.
        p_rand = 0.06
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < p_rand:
                    G.add_edge(a, b)
        n_hubs = int(rng.integers(3, 5))
        hubs = rng.choice(n, size=n_hubs, replace=False)
        for h in hubs:
            for b in range(n):
                if b != h and rng.random() < 0.5:
                    G.add_edge(int(h), int(b))

    # ensure no isolated nodes (add a random edge if needed)
    for node in list(G.nodes()):
        if G.degree(node) == 0:
            other = int(rng.integers(0, n))
            if other != node:
                G.add_edge(node, other)

    edges = np.array(list(G.edges())).T
    if edges.size == 0:
        edges = np.array([[0], [0]])
    edge_index = np.concatenate([edges, edges[::-1]], axis=1)

    return x, edge_index, n


def graph_feature_vector(x):
    """Aggregated node-feature stats per graph (NO topology) for baselines."""
    return np.concatenate([
        x.mean(0), x.std(0),
        np.quantile(x, 0.10, axis=0),
        np.quantile(x, 0.50, axis=0),
        np.quantile(x, 0.90, axis=0),
        x.max(0), x.min(0),
    ]).astype(np.float32)


def main():
    C.set_seed(0)
    rng = np.random.default_rng(0)

    n_graphs = 800
    datas, Xbase, ys = [], [], []
    for i in range(n_graphs):
        disease = i % 2 == 0
        x, edge_index, n = make_graph(rng, disease)
        y = int(disease)
        datas.append(Data(
            x=torch.tensor(x),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor([y], dtype=torch.long),
        ))
        Xbase.append(graph_feature_vector(x))
        ys.append(y)

    Xbase = np.stack(Xbase)
    ys = np.array(ys)
    print(f"graphs={n_graphs}  feat_dim(node)={datas[0].x.size(1)}  "
          f"base_dim={Xbase.shape[1]}  prevalence={ys.mean():.3f}")

    # split
    perm = rng.permutation(n_graphs)
    n_tr, n_va = int(0.6 * n_graphs), int(0.2 * n_graphs)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    # baselines: aggregated node-feature stats only
    xgb = C.run_xgboost(Xbase[tr], ys[tr], Xbase[te], ys[te], num_class=2)
    mlp = C.run_mlp(Xbase[tr], ys[tr], Xbase[te], ys[te], num_class=2)

    # GNN: reads co-occurrence structure
    tr_loader = DataLoader([datas[i] for i in tr], batch_size=32, shuffle=True)
    va_loader = DataLoader([datas[i] for i in va], batch_size=64)
    te_loader = DataLoader([datas[i] for i in te], batch_size=64)
    gnn = C.run_gnn_graph(tr_loader, va_loader, te_loader,
                          in_dim=datas[0].x.size(1), num_class=2, epochs=80)

    C.log_result(
        "exp06_microbiome", "graph-cls", gnn, mlp, xgb, primary="auc",
        notes="Healthy=modular/assortative co-occurrence vs disease=hub-reorganized; "
              "per-taxon abundance marginals identical across classes, so only "
              "co-occurrence topology separates them.",
    )


if __name__ == "__main__":
    main()
