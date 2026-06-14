import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
import common as C
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

"""ReadmitGraph -- 30-day readmission via a patient-similarity graph (node-cls).

Each patient has a full diagnosis/comorbidity vector drawn from one of several
latent clinical communities (e.g. cardiometabolic, oncologic, ...). We build a
patient-similarity graph with kNN on cosine similarity of the FULL diagnosis
vectors, so patients with overlapping conditions become neighbors -- the edges
therefore encode the latent community structure. 30-day readmission risk is
homophilous along this graph: it is driven by the patient's community.

The fair-comparison twist: baselines and the GNN both receive only a NOISY
SUBSET of clinical variables as node features -- too few/too noisy to recover a
patient's community on their own. The kNN edges (built from the full diagnosis
vector) carry the community signal that the node features omit, so only the GNN,
by aggregating over similar patients, can recover community and thus risk.
"""

SEED = 0


def main():
    C.set_seed(SEED)
    rng = np.random.default_rng(SEED)
    n = 2500
    D = 40            # full diagnosis/comorbidity dimensionality
    n_comm = 10       # latent clinical communities
    k = 12            # kNN neighbors

    # Latent community diagnosis profiles (sparse, non-negative "comorbidity load").
    profiles = np.zeros((n_comm, D))
    for c in range(n_comm):
        active = rng.choice(D, size=6, replace=False)
        profiles[c, active] = rng.uniform(0.8, 1.6, size=active.size)

    comm = rng.integers(0, n_comm, size=n)
    full_diag = profiles[comm] + 0.45 * rng.standard_normal((n, D))
    full_diag = np.clip(full_diag, 0, None)

    # Patient-similarity graph: kNN on cosine similarity of the FULL diagnosis vec.
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(full_diag)
    _, knn = nbrs.kneighbors(full_diag)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in knn[i, 1:]:
            G.add_edge(i, int(j))

    # Community-driven readmission risk (homophilous along the similarity graph),
    # with mild within-community variation so it isn't a trivial 1:1 with community.
    comm_risk = rng.uniform(0.0, 1.0, size=n_comm)
    latent = comm_risk[comm] + 0.25 * rng.standard_normal(n)
    y = (latent > np.median(latent)).astype(np.int64)

    # Node features: a NOISY SUBSET of clinical variables (6 of 40 diagnosis dims)
    # plus age -- individually insufficient to identify community or risk.
    subset = rng.choice(D, size=6, replace=False)
    Xsub = full_diag[:, subset] + 0.9 * rng.standard_normal((n, subset.size))
    age = rng.normal(60, 12, n).reshape(-1, 1)
    X = np.concatenate([Xsub, age], axis=1).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    print(f"readmit rate = {y.mean():.3f}  (n={n}, edges={G.number_of_edges()})")

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

    C.log_result("exp02_readmission", "node-cls", gnn, mlp, xgb, primary="auc",
                 notes="kNN patient-similarity graph on full diagnosis vectors; "
                       "readmission homophilous by latent community while node "
                       "features expose only a noisy diagnosis subset.")


if __name__ == "__main__":
    main()
