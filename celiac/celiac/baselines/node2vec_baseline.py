"""
Node2Vec baseline for heterogeneous knowledge graphs.

Adapts node2vec for heterogeneous graphs by:
1. Converting to homogeneous graph (ignoring types)
2. Running random walks
3. Training skip-gram model
4. Using embeddings for link prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import Node2Vec
from typing import Dict, List, Optional, Tuple
import numpy as np


class Node2VecModel(nn.Module):
    """
    Node2Vec baseline for heterogeneous graphs.

    Converts hetero graph to homogeneous and learns embeddings via random walks.
    """

    def __init__(
        self,
        data: HeteroData,
        embedding_dim: int = 64,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
    ):
        """
        Args:
            data: PyG HeteroData object
            embedding_dim: Dimension of embeddings
            walk_length: Length of random walks
            context_size: Context window size for skip-gram
            walks_per_node: Number of walks per node
            p: Return parameter (1/p = probability of returning)
            q: In-out parameter (1/q = probability of going outward)
            num_negative_samples: Number of negative samples per positive
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        # Store metadata
        self.node_types = list(data.node_types)
        self.edge_types = list(data.edge_types)

        # Build node type offsets for converting to homogeneous
        self.node_type_offset = {}
        self.node_type_range = {}
        offset = 0
        for node_type in self.node_types:
            if hasattr(data[node_type], 'num_nodes'):
                num_nodes = data[node_type].num_nodes
            elif hasattr(data[node_type], 'x'):
                num_nodes = data[node_type].x.size(0)
            else:
                # Infer from edges
                num_nodes = 0
                for edge_type in self.edge_types:
                    if edge_type[0] == node_type:
                        num_nodes = max(num_nodes, data[edge_type].edge_index[0].max().item() + 1)
                    if edge_type[2] == node_type:
                        num_nodes = max(num_nodes, data[edge_type].edge_index[1].max().item() + 1)

            self.node_type_offset[node_type] = offset
            self.node_type_range[node_type] = (offset, offset + num_nodes)
            offset += num_nodes

        self.total_nodes = offset

        # Convert to homogeneous graph
        self.homo_edge_index = self._to_homogeneous(data)

        # Node2Vec model (will be initialized during training)
        self.node2vec = None
        self.embeddings = None

    def _to_homogeneous(self, data: HeteroData) -> torch.Tensor:
        """Convert heterogeneous graph to homogeneous edge index."""
        all_edges = []

        for edge_type in self.edge_types:
            if not hasattr(data[edge_type], 'edge_index'):
                continue

            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index

            # Apply offsets
            src_offset = self.node_type_offset[src_type]
            dst_offset = self.node_type_offset[dst_type]

            src = edge_index[0] + src_offset
            dst = edge_index[1] + dst_offset

            all_edges.append(torch.stack([src, dst], dim=0))

            # Add reverse edges for undirected walks
            all_edges.append(torch.stack([dst, src], dim=0))

        if all_edges:
            return torch.cat(all_edges, dim=1)
        else:
            return torch.zeros((2, 0), dtype=torch.long)

    def train_embeddings(
        self,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 0.01,
        device: str = 'cpu',
    ) -> None:
        """
        Train node2vec embeddings.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            device: Device to train on
        """
        edge_index = self.homo_edge_index.to(device)

        # Create Node2Vec model
        self.node2vec = Node2Vec(
            edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            p=self.p,
            q=self.q,
            num_negative_samples=self.num_negative_samples,
            num_nodes=self.total_nodes,
        ).to(device)

        # Training
        loader = self.node2vec.loader(batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.node2vec.parameters(), lr=lr)

        self.node2vec.train()
        for epoch in range(epochs):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.node2vec.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Store embeddings
        self.node2vec.eval()
        with torch.no_grad():
            self.embeddings = self.node2vec().detach()

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Get node embeddings for each type.

        Args:
            data: HeteroData (not used, embeddings are pre-computed)

        Returns:
            Dict mapping node type -> embeddings
        """
        if self.embeddings is None:
            raise RuntimeError("Must call train_embeddings() first")

        z_dict = {}
        for node_type in self.node_types:
            start, end = self.node_type_range[node_type]
            z_dict[node_type] = self.embeddings[start:end]

        return z_dict

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


class Node2VecMLP(nn.Module):
    """
    Node2Vec with MLP decoder for link prediction.

    Combines node2vec embeddings with a learnable MLP for scoring.
    """

    def __init__(
        self,
        data: HeteroData,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
    ):
        super().__init__()

        self.node2vec_model = Node2VecModel(
            data=data,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
        )

        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

        self.edge_types = list(data.edge_types)

    def train_embeddings(self, **kwargs):
        """Train node2vec embeddings."""
        self.node2vec_model.train_embeddings(**kwargs)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Get node embeddings."""
        return self.node2vec_model.forward(data)

    def decode(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_type: Tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode edges using MLP."""
        src_type, _, dst_type = edge_type
        src_z = z_dict[src_type][edge_index[0]]
        dst_z = z_dict[dst_type][edge_index[1]]

        # Concatenate and pass through MLP
        combined = torch.cat([src_z, dst_z], dim=-1)
        return self.decoder(combined).squeeze(-1)
