"""
Models for the inductive vs transductive comparison.

A single ``HeteroLinkModel`` spans a clean 2x2 design so the comparison isolates
exactly two factors with everything else held fixed (same decoder, same depth,
same training loop):

                    | no message passing | message passing (GNN)
  --------------------------------------------------------------
  content features  | feature_mlp        | feature_gnn  (ours)
  learned id embeds | embed_lookup       | embed_gnn

- ``feature_*`` models project shared text features ``data[nt].x`` and therefore
  generalise to nodes unseen at training time.
- ``embed_*`` models use a per-node ``nn.Embedding`` keyed by index; they have no
  trained representation for unseen nodes and collapse on the inductive split.
- ``*_gnn`` models add heterogeneous message passing; ``*_mlp``/``*_lookup`` do
  not, isolating the contribution of graph structure.

All variants share an identical MLP edge decoder, so any performance difference
is attributable to the encoder (features vs ids, graph vs no graph).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

EdgeType = Tuple[str, str, str]

MODES = ("feature_gnn", "feature_mlp", "embed_gnn", "embed_lookup")


class EdgeMLPDecoder(nn.Module):
    """Score an edge from the concatenation of its endpoint representations."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)


class HeteroLinkModel(nn.Module):
    """Unified link-prediction model for the 2x2 comparison."""

    def __init__(
        self,
        data: HeteroData,
        target_edge_type: EdgeType,
        mode: str = "feature_gnn",
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode}")

        self.mode = mode
        self.use_features = mode.startswith("feature")
        self.use_graph = mode.endswith("gnn")
        self.target_edge_type = target_edge_type
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_types: List[str] = list(data.node_types)
        self.edge_types: List[EdgeType] = list(data.edge_types)

        self.num_nodes_dict = {nt: int(data[nt].num_nodes) for nt in self.node_types}

        # Input encoders: feature projection or learned id embeddings.
        if self.use_features:
            self.input_proj = nn.ModuleDict({
                nt: nn.Linear(int(data[nt].x.size(1)), hidden_channels)
                for nt in self.node_types
            })
            self.node_embeddings = None
        else:
            self.input_proj = None
            self.node_embeddings = nn.ModuleDict({
                nt: nn.Embedding(self.num_nodes_dict[nt], hidden_channels)
                for nt in self.node_types
            })
            for emb in self.node_embeddings.values():
                nn.init.xavier_uniform_(emb.weight)

        # Optional message passing.
        self.convs = nn.ModuleList()
        if self.use_graph:
            for _ in range(num_layers):
                conv = HeteroConv(
                    {et: SAGEConv(hidden_channels, hidden_channels) for et in self.edge_types},
                    aggr="sum",
                )
                self.convs.append(conv)

        self.decoder = EdgeMLPDecoder(hidden_channels)

    def _input_x(self, data: HeteroData, device: torch.device) -> Dict[str, torch.Tensor]:
        x_dict = {}
        for nt in self.node_types:
            if self.use_features:
                x_dict[nt] = self.input_proj[nt](data[nt].x.to(device))
            else:
                idx = torch.arange(self.num_nodes_dict[nt], device=device)
                x_dict[nt] = self.node_embeddings[nt](idx)
        return x_dict

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        x_dict = self._input_x(data, device)

        if not self.use_graph:
            return {nt: F.relu(x) for nt, x in x_dict.items()}

        edge_index_dict = {
            et: data[et].edge_index.to(device)
            for et in self.edge_types
            if et in data.edge_types and data[et].edge_index.numel() > 0
        }

        for conv in self.convs:
            out = conv(x_dict, edge_index_dict)
            # Preserve representations for node types that received no messages.
            for nt in x_dict:
                if nt not in out:
                    out[nt] = x_dict[nt]
            x_dict = {nt: F.relu(v) for nt, v in out.items()}
            x_dict = {nt: F.dropout(v, p=self.dropout, training=self.training) for nt, v in x_dict.items()}
        return x_dict

    def decode(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_type: EdgeType,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src_type, _, dst_type = edge_type
        device = z_dict[src_type].device
        edge_index = edge_index.to(device)
        z_src = z_dict[src_type][edge_index[0]]
        z_dst = z_dict[dst_type][edge_index[1]]
        return self.decoder(z_src, z_dst)


# Friendly display names for the four configurations.
MODE_DISPLAY = {
    "feature_gnn": "Feature-GNN (ours)",
    "feature_mlp": "Feature-MLP (no graph)",
    "embed_gnn": "Embed-GNN",
    "embed_lookup": "Embed-Lookup (no graph)",
}
