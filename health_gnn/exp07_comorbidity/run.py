import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
from torch_geometric.data import Data
import common as C

"""
ComorbidityProg: chronic-disease progression on a comorbidity-similarity graph
(node classification).

Story
-----
Patients are linked when their comorbidity profiles overlap, so the graph is a
comorbidity-similarity network whose communities correspond to latent disease
TRAJECTORIES. Whether a patient will progress to a severe stage is governed by
which trajectory community they belong to (and their neighborhood within it) --
progression is homophilous: patients adjacent in comorbidity space share
outcomes.

Each node carries a few noisy clinical markers that are individually
insufficient to call progression (high observation noise). Baselines see only a
patient's own markers and stay weak. The GNN aggregates over the comorbidity
neighborhood, averaging away marker noise across same-trajectory neighbors, and
recovers the progression signal -> wins.
"""


def build_dataset(seed=0):
    rng = np.random.default_rng(seed)

    K = 30                      # latent disease-trajectory communities
    per = 84
    N = K * per                 # 2520 patients
    comm = np.repeat(np.arange(K), per)

    # ---- latent trajectory severity ------------------------------------- #
    # community-level severity dominates; small individual deviation.
    c_sev = rng.normal(0, 1, K)
    indiv = rng.normal(0, 1, N)
    z = c_sev[comm] + 0.55 * indiv          # latent progression score

    # progression label: top ~half of latent score (balanced, homophilous
    # because z is dominated by the shared community-level severity).
    y = (z > np.median(z)).astype(np.int64)

    # ---- comorbidity-similarity graph (SBM over trajectories) ----------- #
    G = nx.Graph()
    G.add_nodes_from(range(N))

    p_in = 0.12
    for k in range(K):
        idx = np.where(comm == k)[0]
        m = len(idx)
        tri = np.triu(rng.random((m, m)) < p_in, k=1)
        a, b = np.where(tri)
        G.add_edges_from(zip(idx[a], idx[b]))

    # sparse cross-trajectory comorbidity overlaps
    n_cross = int(0.6 * N)
    for _ in range(n_cross):
        u, v = rng.integers(0, N, size=2)
        if comm[u] != comm[v]:
            G.add_edge(int(u), int(v))

    # ---- clinical markers: noisy, individually insufficient ------------- #
    # each marker is a noisy peek at the latent progression score z.
    age = rng.normal(60, 14, N)
    m1 = z + rng.normal(0, 2.0, N)
    m2 = z + rng.normal(0, 2.0, N)
    m3 = 0.6 * z + rng.normal(0, 2.2, N)

    X = np.column_stack([
        (age - age.mean()) / age.std(),
        (m1 - m1.mean()) / m1.std(),
        (m2 - m2.mean()) / m2.std(),
        (m3 - m3.mean()) / m3.std(),
    ]).astype(np.float32)

    edges = np.array(list(G.edges())).T
    edge_index = np.concatenate([edges, edges[::-1]], axis=1)

    return X, y, edge_index, G


def main():
    C.set_seed(0)
    X, y, edge_index, G = build_dataset(seed=0)
    N = X.shape[0]
    print(f"N={N}  edges={edge_index.shape[1]//2}  prevalence={y.mean():.3f}  "
          f"avg_deg={2*G.number_of_edges()/N:.1f}")

    rng = np.random.default_rng(1)
    perm = rng.permutation(N)
    n_tr, n_va = int(0.6 * N), int(0.2 * N)
    tr_idx, va_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    def mask(idx):
        m = np.zeros(N, dtype=bool)
        m[idx] = True
        return torch.tensor(m)

    train_mask, val_mask, test_mask = mask(tr_idx), mask(va_idx), mask(te_idx)

    xgb = C.run_xgboost(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], num_class=2)
    mlp = C.run_mlp(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], num_class=2)

    data = Data(
        x=torch.tensor(X),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y),
    )
    gnn = C.run_gnn_node(data, train_mask, val_mask, test_mask, num_class=2, epochs=250)

    C.log_result(
        "exp07_comorbidity", "node-cls", gnn, mlp, xgb, primary="auc",
        notes="Progression label is homophilous over latent disease-trajectory "
              "communities in a comorbidity-similarity graph; per-patient markers "
              "are noisy peeks at the latent score, insufficient on their own.",
    )


if __name__ == "__main__":
    main()
