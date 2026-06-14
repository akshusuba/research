import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
import common as C
from torch_geometric.data import Data

torch.set_num_threads(8)

# ---------------------------------------------------------------------------
# MedKG-Dx: diagnosis prediction over a symptom-disease knowledge graph (node-cls)
#
# ONE heterogeneous-ish graph holds PATIENT nodes and SYMPTOM/concept nodes.
# Patient--symptom edges connect a patient to the symptoms they present; related
# symptoms within a clinical concept community are also linked (symptom--symptom).
#
# A patient's diagnosis (K diseases) is determined RELATIONALLY by WHICH symptom
# community they predominantly connect to. Patient node features are noisy
# demographics that are individually weak (heavy class overlap). Symptom nodes
# carry a concept-community embedding but live in a disjoint feature block, so
# patient feature rows expose no community info.
#
# Baselines see ONLY patient demographic features -> weak. The GNN propagates
# concept information from the symptom nodes a patient links to -> wins.
# ---------------------------------------------------------------------------

K = 4                 # number of diseases / symptom communities
SYM_PER_COMM = 30     # symptom nodes per community
N_PATIENTS = 2400
D_DEMO = 8            # noisy demographic features (weak)
DEMO_SCALE = 0.45     # strength of (weak) demographic signal
IN_DIM = D_DEMO + K   # demographics block + symptom-community block


def build():
    n_sym = K * SYM_PER_COMM
    sym_comm = np.repeat(np.arange(K), SYM_PER_COMM)        # community of each symptom
    n_total = N_PATIENTS + n_sym

    # ---- labels & demographics for patients ----
    y_pat = np.random.randint(0, K, size=N_PATIENTS)
    centroids = np.random.randn(K, D_DEMO)
    demo = DEMO_SCALE * centroids[y_pat] + np.random.randn(N_PATIENTS, D_DEMO)

    # ---- node feature matrix (shared) ----
    x = np.zeros((n_total, IN_DIM), dtype=np.float32)
    x[:N_PATIENTS, :D_DEMO] = demo.astype(np.float32)
    # symptom nodes: concept-community embedding in the disjoint block + noise
    sym_oh = np.zeros((n_sym, K), dtype=np.float32)
    sym_oh[np.arange(n_sym), sym_comm] = 1.0
    x[N_PATIENTS:, D_DEMO:] = sym_oh + 0.15 * np.random.randn(n_sym, K).astype(np.float32)

    # ---- edges ----
    edges = []
    # symptom--symptom edges within a community (clinical concept relatedness)
    for c in range(K):
        members = np.where(sym_comm == c)[0] + N_PATIENTS
        for _ in range(SYM_PER_COMM):  # ~1 edge per member
            a, b = np.random.choice(members, 2, replace=False)
            edges.append((int(a), int(b)))

    # patient--symptom edges: dominated by the patient's own community, with noise
    for p in range(N_PATIENTS):
        k = y_pat[p]
        n_sym_p = np.random.randint(5, 10)
        for _ in range(n_sym_p):
            if np.random.rand() < 0.78:
                comm = k
            else:
                comm = np.random.randint(0, K)
            s = np.random.randint(0, SYM_PER_COMM) + comm * SYM_PER_COMM + N_PATIENTS
            edges.append((p, int(s)))

    e = np.array(edges, dtype=np.int64).T
    ei = np.concatenate([e, e[::-1]], axis=1)  # undirected

    # ---- labels tensor (symptom nodes get a dummy 0, never evaluated) ----
    y = np.zeros(n_total, dtype=np.int64)
    y[:N_PATIENTS] = y_pat

    # ---- masks over PATIENT nodes only ----
    idx = np.random.permutation(N_PATIENTS)
    n_tr, n_va = int(0.6 * N_PATIENTS), int(0.2 * N_PATIENTS)
    tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[torch.tensor(tr_idx)] = True
    val_mask[torch.tensor(va_idx)] = True
    test_mask[torch.tensor(te_idx)] = True

    data = Data(x=torch.tensor(x), edge_index=torch.tensor(ei), y=torch.tensor(y))
    return data, train_mask, val_mask, test_mask, demo, y_pat, (tr_idx, va_idx, te_idx)


def main():
    C.set_seed(0)
    data, train_mask, val_mask, test_mask, demo, y_pat, (tr, va, te) = build()

    # Baselines: PATIENT demographic features only (fair, no graph info).
    Xtr, ytr = demo[tr], y_pat[tr]
    Xte, yte = demo[te], y_pat[te]
    xgb = C.run_xgboost(Xtr, ytr, Xte, yte, num_class=K)
    mlp = C.run_mlp(Xtr, ytr, Xte, yte, num_class=K)

    gnn = C.run_gnn_node(data, train_mask, val_mask, test_mask, num_class=K, epochs=300)

    C.log_result(
        "exp09_medkg", "node-cls", gnn, mlp, xgb, primary="auc",
        notes=("Single symptom-disease knowledge graph with patient and symptom/concept "
               "nodes. Diagnosis (4 diseases) is set by which symptom community a patient "
               "connects to. Patient features are weak noisy demographics; symptom nodes "
               "carry concept embeddings in a disjoint feature block. Baselines see patient "
               "demographics only and stay weak; the GNN propagates concept information "
               "across patient-symptom edges and wins."))


if __name__ == "__main__":
    main()
