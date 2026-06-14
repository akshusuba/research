import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
from torch_geometric.data import Data
import common as C

"""
HAI-Transfer: healthcare-associated infection (HAI) risk on a patient-transfer
network (node classification).

Story
-----
Patients/episodes live inside wards of a hospital system. Edges encode shared
ward stays plus inter-ward transfer paths, so the graph is a clustered
transfer network. A pathogen is seeded in a few wards and propagates along
transfer edges via an independent-cascade contagion: colonization is therefore
a *structural* property -- who you are connected to, and whether contagion
reached your part of the network.

Node features are patient attributes (age, comorbidity score, length-of-stay,
two noisy clinical risk markers). Susceptibility depends only weakly on the
comorbidity score, so the features are individually weak predictors of
colonization and carry NO information about transfer topology. Baselines
(XGBoost / MLP) see these node attributes only; the GNN additionally passes
messages along transfer edges and recovers the contagion structure -> wins.
"""


def build_dataset(seed=0):
    rng = np.random.default_rng(seed)

    n_wards = 42
    per_ward = 60
    N = n_wards * per_ward  # 2520 patients

    ward_of = np.repeat(np.arange(n_wards), per_ward)

    # ---- build transfer network ------------------------------------------- #
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # dense intra-ward contact (shared ward stays)
    p_in = 0.14
    for w in range(n_wards):
        idx = np.where(ward_of == w)[0]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if rng.random() < p_in:
                    G.add_edge(idx[a], idx[b])

    # inter-ward transfer edges following a sparse ward-level transfer graph
    ward_net = nx.connected_watts_strogatz_graph(n_wards, 4, 0.3, seed=seed)
    for w1, w2 in ward_net.edges():
        i1 = np.where(ward_of == w1)[0]
        i2 = np.where(ward_of == w2)[0]
        n_transfer = rng.integers(4, 9)
        for _ in range(n_transfer):
            a = rng.choice(i1)
            b = rng.choice(i2)
            G.add_edge(int(a), int(b))

    adj = {i: list(G.neighbors(i)) for i in range(N)}

    # ---- latent susceptibility (drives contagion, only weakly observed) --- #
    # s_i is the true per-patient susceptibility. It nudges who gets colonized,
    # but the dominant driver of colonization is structural exposure (whether
    # the cascade reached you). s_i is observed only through noisy clinical
    # markers, so node features are individually weak predictors of the label.
    s = rng.normal(0.0, 1.0, N)
    suscept = 1.0 / (1.0 + np.exp(-0.6 * s))  # in (0,1), mean ~0.5

    # ---- independent-cascade contagion over transfer edges ---------------- #
    infected = np.zeros(N, dtype=bool)
    seed_wards = rng.choice(n_wards, size=10, replace=False)
    for w in seed_wards:
        idx = np.where(ward_of == w)[0]
        picks = rng.choice(idx, size=4, replace=False)
        infected[picks] = True

    beta = 0.16  # per-contact transmission probability (R0 > 1 -> spreads)
    frontier = list(np.where(infected)[0])
    for _ in range(40):  # cascade rounds (saturates well before this)
        new_frontier = []
        for u in frontier:
            for v in adj[u]:
                if not infected[v] and rng.random() < beta * suscept[v]:
                    infected[v] = True
                    new_frontier.append(v)
        frontier = new_frontier
        if not frontier:
            break

    y = infected.astype(np.int64)

    # ---- patient attributes (weak, NO topology information) --------------- #
    age = rng.normal(65, 16, N)
    los = rng.gamma(2.5, 2.0, N)
    # comorbidity score is a NOISY observation of latent susceptibility s:
    # weakly correlated with colonization, individually far from sufficient.
    comorbidity = s + rng.normal(0.0, 2.2, N)
    # two extra clinical risk markers: very noisy peeks at s
    risk1 = 0.5 * s + rng.normal(0.0, 2.5, N)
    risk2 = 0.5 * s + rng.normal(0.0, 2.5, N)

    X = np.column_stack([
        (age - age.mean()) / age.std(),
        (comorbidity - comorbidity.mean()) / comorbidity.std(),
        (los - los.mean()) / los.std(),
        (risk1 - risk1.mean()) / risk1.std(),
        (risk2 - risk2.mean()) / risk2.std(),
    ]).astype(np.float32)

    edges = np.array(list(G.edges())).T
    edge_index = np.concatenate([edges, edges[::-1]], axis=1)  # undirected

    return X, y, edge_index, G


def main():
    C.set_seed(0)
    X, y, edge_index, G = build_dataset(seed=0)
    N = X.shape[0]
    print(f"N={N}  edges={edge_index.shape[1]//2}  prevalence={y.mean():.3f}  "
          f"avg_deg={2*G.number_of_edges()/N:.1f}")

    # train/val/test split
    rng = np.random.default_rng(1)
    perm = rng.permutation(N)
    n_tr, n_va = int(0.6 * N), int(0.2 * N)
    tr_idx, va_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    def mask(idx):
        m = np.zeros(N, dtype=bool)
        m[idx] = True
        return torch.tensor(m)

    train_mask, val_mask, test_mask = mask(tr_idx), mask(va_idx), mask(te_idx)

    # baselines: node features only
    xgb = C.run_xgboost(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], num_class=2)
    mlp = C.run_mlp(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], num_class=2)

    # GNN: same node features + transfer edges
    data = Data(
        x=torch.tensor(X),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y),
    )
    gnn = C.run_gnn_node(data, train_mask, val_mask, test_mask, num_class=2, epochs=250)

    C.log_result(
        "exp05_hai_transfer", "node-cls", gnn, mlp, xgb, primary="auc",
        notes="HAI colonization via independent-cascade contagion on a clustered "
              "ward/transfer network; node attributes weakly predictive only.",
    )


if __name__ == "__main__":
    main()
