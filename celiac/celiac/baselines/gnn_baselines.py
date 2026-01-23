"""
GNN baseline models for heterogeneous knowledge graphs.

Implements:
- R-GCN: Relational Graph Convolutional Networks (Schlichtkrull et al., 2018)
- CompGCN: Composition-based GCN (Vashishth et al., 2020)
- HGT: Heterogeneous Graph Transformer (Hu et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    RGCNConv,
    HGTConv,
    HeteroConv,
    Linear,
)
from typing import Dict, List, Optional, Tuple, Union
import math


class RGCNModel(nn.Module):
    """
    Relational Graph Convolutional Network for heterogeneous graphs.

    Uses relation-specific weight matrices for message passing.
    """

    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_bases: Optional[int] = None,
    ):
        """
        Args:
            data: PyG HeteroData object
            hidden_channels: Hidden dimension
            num_layers: Number of RGCN layers
            dropout: Dropout rate
            num_bases: Number of basis matrices for decomposition (None = no decomposition)
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Get metadata
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        num_relations = len(self.edge_types)

        # Node type to number of nodes
        self.num_nodes_dict = {}
        for node_type in self.node_types:
            if hasattr(data[node_type], 'num_nodes'):
                self.num_nodes_dict[node_type] = data[node_type].num_nodes
            elif hasattr(data[node_type], 'x'):
                self.num_nodes_dict[node_type] = data[node_type].x.size(0)

        # Learnable node embeddings per type
        self.node_embeddings = nn.ModuleDict()
        for node_type, num_nodes in self.num_nodes_dict.items():
            self.node_embeddings[node_type] = nn.Embedding(num_nodes, hidden_channels)

        # If num_bases is None, use min of num_relations and hidden_channels // 4
        if num_bases is None:
            num_bases = min(num_relations, max(1, hidden_channels // 4))

        # RGCN layers using HeteroConv wrapper
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                src, rel, dst = edge_type
                conv_dict[edge_type] = RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations=1,  # Each conv handles one relation
                    num_bases=1,
                    aggr='mean',
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings."""
        for emb in self.node_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get node embeddings.

        Args:
            data: HeteroData object

        Returns:
            Dict mapping node type -> embeddings
        """
        # Initialize embeddings
        x_dict = {}
        for node_type in self.node_types:
            num_nodes = self.num_nodes_dict[node_type]
            indices = torch.arange(num_nodes, device=next(self.parameters()).device)
            x_dict[node_type] = self.node_embeddings[node_type](indices)

        # Get edge indices
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in self.edge_types
            if hasattr(data[edge_type], 'edge_index')
        }

        # Apply RGCN layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        return x_dict

    def decode(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_type: Tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode edges using dot product."""
        src_type, _, dst_type = edge_type
        src_z = z_dict[src_type][edge_index[0]]
        dst_z = z_dict[dst_type][edge_index[1]]
        return (src_z * dst_z).sum(dim=-1)


class CompGCNModel(nn.Module):
    """
    Composition-based Graph Convolutional Network.

    Jointly embeds nodes and relations using composition operations.
    """

    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        composition: str = 'mult',  # 'mult', 'sub', or 'corr'
    ):
        """
        Args:
            data: PyG HeteroData object
            hidden_channels: Hidden dimension
            num_layers: Number of CompGCN layers
            dropout: Dropout rate
            composition: Composition operation ('mult', 'sub', 'corr')
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.composition = composition

        # Get metadata
        self.node_types = data.node_types
        self.edge_types = data.edge_types

        # Node counts
        self.num_nodes_dict = {}
        for node_type in self.node_types:
            if hasattr(data[node_type], 'num_nodes'):
                self.num_nodes_dict[node_type] = data[node_type].num_nodes
            elif hasattr(data[node_type], 'x'):
                self.num_nodes_dict[node_type] = data[node_type].x.size(0)

        # Node embeddings
        self.node_embeddings = nn.ModuleDict()
        for node_type, num_nodes in self.num_nodes_dict.items():
            self.node_embeddings[node_type] = nn.Embedding(num_nodes, hidden_channels)

        # Relation embeddings (one per edge type)
        self.relation_embeddings = nn.Embedding(len(self.edge_types), hidden_channels)
        self.edge_type_to_idx = {et: i for i, et in enumerate(self.edge_types)}

        # CompGCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            # Weight matrices for composition
            self.convs.append(nn.ModuleDict({
                node_type: nn.Linear(hidden_channels, hidden_channels, bias=False)
                for node_type in self.node_types
            }))
            self.norms.append(nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_channels)
                for node_type in self.node_types
            }))

        # Relation transformation per layer
        self.rel_transforms = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for emb in self.node_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def compose(
        self,
        node_emb: torch.Tensor,
        rel_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply composition operation."""
        if self.composition == 'mult':
            return node_emb * rel_emb
        elif self.composition == 'sub':
            return node_emb - rel_emb
        elif self.composition == 'corr':
            # Circular correlation
            node_fft = torch.fft.rfft(node_emb, dim=-1)
            rel_fft = torch.fft.rfft(rel_emb, dim=-1)
            return torch.fft.irfft(node_fft * torch.conj(rel_fft), n=node_emb.size(-1), dim=-1)
        else:
            raise ValueError(f"Unknown composition: {self.composition}")

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        device = next(self.parameters()).device

        # Initialize embeddings
        x_dict = {}
        for node_type in self.node_types:
            num_nodes = self.num_nodes_dict[node_type]
            indices = torch.arange(num_nodes, device=device)
            x_dict[node_type] = self.node_embeddings[node_type](indices)

        rel_emb = self.relation_embeddings.weight

        # Apply CompGCN layers
        for layer_idx in range(self.num_layers):
            new_x_dict = {nt: torch.zeros_like(x_dict[nt]) for nt in self.node_types}
            counts = {nt: 0 for nt in self.node_types}

            for edge_type in self.edge_types:
                src_type, _, dst_type = edge_type
                if not hasattr(data[edge_type], 'edge_index'):
                    continue

                edge_index = data[edge_type].edge_index.to(device)
                rel_idx = self.edge_type_to_idx[edge_type]
                r = rel_emb[rel_idx]

                # Compose source node with relation
                src_emb = x_dict[src_type][edge_index[0]]
                composed = self.compose(src_emb, r.unsqueeze(0).expand_as(src_emb))

                # Aggregate to destination
                new_x_dict[dst_type].index_add_(0, edge_index[1], composed)
                counts[dst_type] += 1

            # Normalize and transform
            for node_type in self.node_types:
                if counts[node_type] > 0:
                    x = new_x_dict[node_type] / max(counts[node_type], 1)
                    x = self.convs[layer_idx][node_type](x)
                    x = self.norms[layer_idx][node_type](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x_dict[node_type] = x + x_dict[node_type]  # Residual

            # Update relation embeddings
            rel_emb = self.rel_transforms[layer_idx](rel_emb)

        return x_dict

    def decode(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_type: Tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode edges using composition-based scoring."""
        src_type, _, dst_type = edge_type
        rel_idx = self.edge_type_to_idx[edge_type]
        rel_emb = self.relation_embeddings.weight[rel_idx]

        src_z = z_dict[src_type][edge_index[0]]
        dst_z = z_dict[dst_type][edge_index[1]]

        # Score using composition
        composed = self.compose(src_z, rel_emb.unsqueeze(0).expand_as(src_z))
        return (composed * dst_z).sum(dim=-1)


class HGTModel(nn.Module):
    """
    Heterogeneous Graph Transformer.

    Uses type-specific attention mechanisms for heterogeneous graphs.
    """

    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            data: PyG HeteroData object
            hidden_channels: Hidden dimension
            num_layers: Number of HGT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Get metadata
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        metadata = (self.node_types, self.edge_types)

        # Node counts
        self.num_nodes_dict = {}
        for node_type in self.node_types:
            if hasattr(data[node_type], 'num_nodes'):
                self.num_nodes_dict[node_type] = data[node_type].num_nodes
            elif hasattr(data[node_type], 'x'):
                self.num_nodes_dict[node_type] = data[node_type].x.size(0)

        # Node embeddings
        self.node_embeddings = nn.ModuleDict()
        for node_type, num_nodes in self.num_nodes_dict.items():
            self.node_embeddings[node_type] = nn.Embedding(num_nodes, hidden_channels)

        # HGT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
            ))

        # Output projections
        self.out_proj = nn.ModuleDict()
        for node_type in self.node_types:
            self.out_proj[node_type] = nn.Linear(hidden_channels, hidden_channels)

        self._init_weights()

    def _init_weights(self):
        for emb in self.node_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        device = next(self.parameters()).device

        # Initialize embeddings
        x_dict = {}
        for node_type in self.node_types:
            num_nodes = self.num_nodes_dict[node_type]
            indices = torch.arange(num_nodes, device=device)
            x_dict[node_type] = self.node_embeddings[node_type](indices)

        # Get edge indices
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = data[edge_type].edge_index.to(device)

        # Apply HGT layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        # Output projection
        x_dict = {k: self.out_proj[k](v) for k, v in x_dict.items()}

        return x_dict

    def decode(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_type: Tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode edges using dot product."""
        src_type, _, dst_type = edge_type
        src_z = z_dict[src_type][edge_index[0]]
        dst_z = z_dict[dst_type][edge_index[1]]
        return (src_z * dst_z).sum(dim=-1)
