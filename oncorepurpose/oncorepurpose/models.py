"""Models for drug->disease link prediction on PrimeKG (all PyTorch Geometric).

- HeteroGNN: feature-projection encoder + heterogeneous SAGE message passing.
  Consumes shared node features (data[type].x), so it generalizes to nodes unseen
  at training time (inductive). Trained with neighbor sampling; evaluated with a
  single full-graph forward.
- FeatureMLP: same feature projection, NO message passing -- a neural structure-
  blind control (companion to the XGBoost tabular baseline).
- DistMultKGE: per-entity learnable embeddings + DistMult scoring -- a memorization
  baseline that has no representation for unseen nodes and collapses on cold splits.

All scorers share an EdgeMLPDecoder so the GNN-vs-MLP gap isolates message passing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

EdgeType = Tuple[str, str, str]


class EdgeMLPDecoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)


class MechanismHead(nn.Module):
    """Scores whether gene g is the bridge explaining (drug d -> disease c).

    Operates on the SAME GNN node embeddings used for link prediction, so any gain
    on mechanism recovery is attributable to graph structure, not extra features.
    Tabular models (XGBoost) have no analogue: they never embed a third node.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, z_drug: torch.Tensor, z_gene: torch.Tensor, z_dis: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_drug, z_gene, z_dis], dim=-1)).squeeze(-1)


class HeteroGNN(nn.Module):
    """Feature-based heterogeneous GraphSAGE encoder + MLP edge decoder.

    Optionally carries a MechanismHead (used only by the joint mechanism-recovery
    experiment); models that never call ``score_mechanism`` are unaffected.
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[EdgeType],
        in_dims: Dict[str, int],
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout

        self.proj = nn.ModuleDict({nt: nn.Linear(in_dims[nt], hidden) for nt in node_types})
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv({et: SAGEConv(hidden, hidden) for et in edge_types}, aggr="sum")
            )
        self.decoder = EdgeMLPDecoder(hidden)
        self.mech_head = MechanismHead(hidden)

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        x_dict = {nt: self.proj[nt](data[nt].x.to(device)) for nt in self.node_types if nt in data.node_types}
        edge_index_dict = {
            et: data[et].edge_index.to(device)
            for et in self.edge_types
            if et in data.edge_types and data[et].edge_index.numel() > 0
        }
        for conv in self.convs:
            out = conv(x_dict, edge_index_dict)
            for nt in x_dict:
                if nt not in out:
                    out[nt] = x_dict[nt]
            x_dict = {nt: F.dropout(F.relu(v), p=self.dropout, training=self.training) for nt, v in out.items()}
        return x_dict

    def decode(self, z_dict: Dict[str, torch.Tensor], et: EdgeType, eli: torch.Tensor) -> torch.Tensor:
        s_t, _, d_t = et
        dev = z_dict[s_t].device
        eli = eli.to(dev)
        return self.decoder(z_dict[s_t][eli[0]], z_dict[d_t][eli[1]])

    def score_mechanism(
        self,
        z_dict: Dict[str, torch.Tensor],
        drug_idx: torch.Tensor,
        gene_idx: torch.Tensor,
        dis_idx: torch.Tensor,
        drug_type: str = "drug",
        gene_type: str = "gene_protein",
        dis_type: str = "disease",
    ) -> torch.Tensor:
        """Score (drug, gene, disease) triples; all index tensors share length."""
        dev = z_dict[drug_type].device
        return self.mech_head(
            z_dict[drug_type][drug_idx.to(dev)],
            z_dict[gene_type][gene_idx.to(dev)],
            z_dict[dis_type][dis_idx.to(dev)],
        )


class FeatureMLP(nn.Module):
    """Structure-blind neural control: project endpoint features, decode, no graph."""

    def __init__(self, node_types: List[str], in_dims: Dict[str, int], hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.node_types = node_types
        self.dropout = dropout
        self.proj = nn.ModuleDict({nt: nn.Linear(in_dims[nt], hidden) for nt in node_types})
        self.decoder = EdgeMLPDecoder(hidden)

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        return {nt: F.relu(self.proj[nt](data[nt].x.to(device))) for nt in self.node_types if nt in data.node_types}

    def decode(self, z_dict: Dict[str, torch.Tensor], et: EdgeType, eli: torch.Tensor) -> torch.Tensor:
        s_t, _, d_t = et
        dev = z_dict[s_t].device
        eli = eli.to(dev)
        return self.decoder(z_dict[s_t][eli[0]], z_dict[d_t][eli[1]])


class DistMultKGE(nn.Module):
    """Per-entity embeddings + DistMult scoring for the target relation.

    Pure memorization: unseen nodes get an untrained embedding -> collapses on
    cold (inductive) splits, which is exactly the contrast we want to show.
    """

    def __init__(self, src_type: str, dst_type: str, num_src: int, num_dst: int, dim: int = 128):
        super().__init__()
        self.src_type, self.dst_type = src_type, dst_type
        self.src_emb = nn.Embedding(num_src, dim)
        self.dst_emb = nn.Embedding(num_dst, dim)
        self.rel = nn.Parameter(torch.randn(dim) * 0.1)
        nn.init.xavier_uniform_(self.src_emb.weight)
        nn.init.xavier_uniform_(self.dst_emb.weight)

    def score(self, eli: torch.Tensor) -> torch.Tensor:
        dev = self.rel.device
        eli = eli.to(dev)
        s = self.src_emb(eli[0])
        d = self.dst_emb(eli[1])
        return (s * self.rel * d).sum(-1)
