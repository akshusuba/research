import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
import common as C
from torch_geometric.data import Data

"""EpiGNN -- infection risk on a human contact network (node classification).

A contact graph is built (Watts-Strogatz small-world: high clustering, the way
real human contact networks look). We seed a few infections and run an
SI-style contagion for K steps so that infection spreads ALONG EDGES. The
horizon infection state is the label -- and because contagion spreads through
contact, infected nodes form tight clusters (strong label homophily on the
graph). Node features are individual attributes (age, susceptibility, two noisy
risk scores) that are only weakly predictive on their own and DO NOT encode who
each person contacted. Tabular baselines see only those weak per-node features;
the GNN additionally smooths information across the contact graph, exploiting
the contagion homophily to recover infection risk.
"""

SEED = 0


def build_contact_graph(n, k=10, p=0.12, rng=None):
    seed = int(rng.integers(1 << 30)) if rng is not None else SEED
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G


def si_contagion(G, n_seeds, beta, steps, susceptibility, rng):
    """SI contagion: a susceptible node gets infected with prob
    1-(1-beta*s)^(#infected neighbors) each step, where s is its susceptibility."""
    n = G.number_of_nodes()
    infected = np.zeros(n, dtype=bool)
    seeds = rng.choice(n, size=n_seeds, replace=False)
    infected[seeds] = True
    adj = [list(G.neighbors(i)) for i in range(n)]
    for _ in range(steps):
        new = infected.copy()
        for i in range(n):
            if infected[i]:
                continue
            inf_nb = sum(infected[j] for j in adj[i])
            if inf_nb == 0:
                continue
            prob = 1.0 - (1.0 - beta * susceptibility[i]) ** inf_nb
            if rng.random() < prob:
                new[i] = True
        infected = new
    return infected


def main():
    C.set_seed(SEED)
    rng = np.random.default_rng(SEED)
    n = 2500

    G = build_contact_graph(n, k=10, p=0.12, rng=rng)

    # Individual attributes (weak, do not encode contacts).
    age = rng.normal(50, 15, n)
    susceptibility = np.clip(0.5 + 0.15 * rng.standard_normal(n), 0.05, 0.95)

    y = si_contagion(G, n_seeds=25, beta=0.14, steps=12,
                     susceptibility=susceptibility, rng=rng).astype(np.int64)

    # Weak, noisy individual risk scores correlated with the outcome but far too
    # noisy to classify a single person well. (These are personal risk markers,
    # NOT graph statistics.)
    yc = (2 * y - 1).astype(np.float64)
    risk1 = 0.45 * yc + rng.standard_normal(n)
    risk2 = 0.35 * yc + 1.1 * rng.standard_normal(n)
    age_feat = age + 6.0 * yc  # older people slightly likelier infected, very noisy

    X = np.stack([
        (age_feat - age_feat.mean()) / age_feat.std(),
        (susceptibility - susceptibility.mean()) / susceptibility.std(),
        risk1,
        risk2,
    ], axis=1).astype(np.float32)

    print(f"infected rate = {y.mean():.3f}  (n={n}, edges={G.number_of_edges()})")

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
                         epochs=250)

    C.log_result("exp01_epidemic", "node-cls", gnn, mlp, xgb, primary="auc",
                 notes="SI contagion on Watts-Strogatz contact graph; "
                       "infection homophily lets GNN denoise weak personal risk features.")


if __name__ == "__main__":
    main()
