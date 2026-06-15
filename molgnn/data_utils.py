"""Shared featurization + splitting utilities.

Same molecules feed both models:
  - GNN: RDKit mol -> PyG Data (atom features + bond edges)
  - XGBoost: Morgan/ECFP fingerprint (radius 2, 2048 bits)

Split: Bemis-Murcko scaffold split (rigorous molecular standard).
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

# ---- Atom featurization ----
ALLOWED_ATOMS = [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # B,C,N,O,F,Si,P,S,Cl,Br,I
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _onehot(val, choices):
    vec = [0] * (len(choices) + 1)
    if val in choices:
        vec[choices.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom):
    feats = []
    feats += _onehot(atom.GetAtomicNum(), ALLOWED_ATOMS)
    feats += _onehot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    feats += _onehot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feats += _onehot(atom.GetHybridization(), HYBRIDIZATIONS)
    feats += _onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feats += [int(atom.GetIsAromatic())]
    feats += [int(atom.IsInRing())]
    feats += [atom.GetMass() * 0.01]
    return feats


ATOM_FDIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def mol_to_graph(mol, y):
    """RDKit mol -> PyG Data with atom features and bidirectional bond edges."""
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float))


def morgan_fp(mol, radius=2, n_bits=2048):
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    arr = gen.GetFingerprintAsNumPy(mol)
    return arr.astype(np.float32)


def bemis_murcko_scaffold(smiles, include_chirality=False):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    except Exception:
        return ""


def load_dataset(csv_path, threshold=6.5):
    """Load CSV -> parsed molecules with graphs, fingerprints, scaffolds, labels.

    Returns a dict of aligned lists/arrays.
    """
    df = pd.read_csv(csv_path)
    graphs, fps, scaffolds, labels, smiles_kept = [], [], [], [], []
    n_bad = 0
    for smi, pchembl in zip(df["smiles"], df["pchembl_value"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            n_bad += 1
            continue
        y = int(pchembl >= threshold)
        graphs.append(mol_to_graph(mol, y))
        fps.append(morgan_fp(mol))
        scaffolds.append(bemis_murcko_scaffold(smi))
        labels.append(y)
        smiles_kept.append(smi)
    print(f"Parsed {len(graphs)} molecules ({n_bad} unparseable), threshold={threshold}")
    return {
        "graphs": graphs,
        "fps": np.stack(fps),
        "scaffolds": scaffolds,
        "labels": np.array(labels),
        "smiles": smiles_kept,
    }


def scaffold_split(scaffolds, frac_train=0.8, frac_val=0.1, seed=0):
    """Bemis-Murcko scaffold split.

    Groups molecules by scaffold, then assigns whole scaffold-groups to
    train/val/test. Largest groups go to train first (deterministic ordering),
    and `seed` shuffles the order of equal-sized groups for cross-seed variation.
    """
    scaffold_to_idx = defaultdict(list)
    for i, s in enumerate(scaffolds):
        scaffold_to_idx[s].append(i)

    rng = np.random.RandomState(seed)
    groups = list(scaffold_to_idx.values())
    # Sort by size desc; break ties randomly per seed for variation across seeds
    order = sorted(range(len(groups)),
                   key=lambda k: (len(groups[k]), rng.random()), reverse=True)
    groups = [groups[k] for k in order]

    n = len(scaffolds)
    n_train, n_val = int(frac_train * n), int(frac_val * n)
    train, val, test = [], [], []
    for g in groups:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(val) + len(g) <= n_val:
            val += g
        else:
            test += g
    return np.array(train), np.array(val), np.array(test)


def random_split(n, frac_train=0.8, frac_val=0.1, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train, n_val = int(frac_train * n), int(frac_val * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
