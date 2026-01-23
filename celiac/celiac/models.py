"""
GNN Models for Celiac Gut-Brain Knowledge Graph.
Implements Heterogeneous Graph Neural Networks using PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for link prediction.
    Uses separate convolutions for each edge type.
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        conv_type: str = "sage"  # "sage" or "gat"
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Node embeddings (learnable, since we don't have good initial features)
        self.node_embeddings = nn.ModuleDict()

        # Convolution layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src_type, rel, dst_type = edge_type
                if conv_type == "sage":
                    conv_dict[edge_type] = SAGEConv(
                        hidden_channels, hidden_channels
                    )
                else:  # gat
                    conv_dict[edge_type] = GATConv(
                        hidden_channels, hidden_channels // 4,
                        heads=4, dropout=dropout
                    )

            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # Output projection per node type
        self.out_proj = nn.ModuleDict({
            node_type: Linear(hidden_channels, hidden_channels)
            for node_type in node_types
        })

    def init_embeddings(self, data: HeteroData) -> None:
        """Initialize node embeddings based on data."""
        for node_type in self.node_types:
            if node_type in data.node_types:
                num_nodes = data[node_type].num_nodes
                self.node_embeddings[node_type] = nn.Embedding(
                    num_nodes, self.hidden_channels
                )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns dict of node embeddings per type.
        """
        device = next(self.parameters()).device

        # Get initial embeddings
        x_dict = {}
        for node_type in self.node_types:
            if node_type in self.node_embeddings:
                num_nodes = data[node_type].num_nodes
                x_dict[node_type] = self.node_embeddings[node_type](
                    torch.arange(num_nodes, device=device)
                )

        # Store original embeddings for nodes that might not receive messages
        original_x_dict = {k: v.clone() for k, v in x_dict.items()}

        # Get edge indices - only include edge types that have edges
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                if edge_index is not None and edge_index.numel() > 0:
                    edge_index_dict[edge_type] = edge_index.to(device)

        # Message passing
        for conv in self.convs:
            # Run convolution
            new_x_dict = conv(x_dict, edge_index_dict)

            # Preserve embeddings for node types that didn't receive messages
            for node_type in x_dict:
                if node_type not in new_x_dict:
                    new_x_dict[node_type] = x_dict[node_type]

            x_dict = {k: F.relu(v) for k, v in new_x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                     for k, v in x_dict.items()}

        # Make sure all node types have embeddings (use original if missing)
        for node_type in original_x_dict:
            if node_type not in x_dict:
                x_dict[node_type] = original_x_dict[node_type]

        # Output projection
        out_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.out_proj:
                out_dict[node_type] = self.out_proj[node_type](x)
            else:
                out_dict[node_type] = x

        return out_dict


class LinkPredictor(nn.Module):
    """
    Link prediction decoder using bilinear scoring.
    """

    def __init__(self, hidden_channels: int, num_relations: int = 1):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Bilinear weights per relation type
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_channels, hidden_channels))
            for _ in range(num_relations)
        ])

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        relation_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute link scores.
        Args:
            z_src: Source node embeddings [batch, hidden]
            z_dst: Destination node embeddings [batch, hidden]
            relation_idx: Which relation weight to use
        Returns:
            Scores [batch]
        """
        W = self.weights[relation_idx]
        # Bilinear: src @ W @ dst.T for each pair
        scores = (z_src @ W * z_dst).sum(dim=-1)
        return scores


class CeliacGNN(nn.Module):
    """
    Complete model for celiac gut-brain link prediction.
    Combines encoder (HeteroGNN) and decoder (LinkPredictor).
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        conv_type: str = "sage"
    ):
        super().__init__()

        self.encoder = HeteroGNN(
            node_types=node_types,
            edge_types=edge_types,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type
        )

        # Decoders for different prediction tasks
        self.gene_pheno_decoder = LinkPredictor(hidden_channels, num_relations=1)
        self.microbe_pheno_decoder = LinkPredictor(hidden_channels, num_relations=1)

    def init_embeddings(self, data: HeteroData) -> None:
        """Initialize embeddings from data."""
        self.encoder.init_embeddings(data)

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Encode nodes."""
        return self.encoder(data)

    def decode_gene_phenotype(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """Decode gene-phenotype links."""
        z_gene = z_dict["gene"][edge_label_index[0]]
        z_pheno = z_dict["phenotype"][edge_label_index[1]]
        return self.gene_pheno_decoder(z_gene, z_pheno)

    def decode_microbe_phenotype(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """Decode microbe-phenotype links (indirect, through path)."""
        z_microbe = z_dict["microbe"][edge_label_index[0]]
        z_pheno = z_dict["phenotype"][edge_label_index[1]]
        return self.microbe_pheno_decoder(z_microbe, z_pheno)


def create_model_from_data(data: HeteroData, **kwargs) -> CeliacGNN:
    """
    Create model from HeteroData object.
    """
    node_types = list(data.node_types)
    edge_types = list(data.edge_types)

    model = CeliacGNN(
        node_types=node_types,
        edge_types=edge_types,
        **kwargs
    )
    model.init_embeddings(data)

    return model
