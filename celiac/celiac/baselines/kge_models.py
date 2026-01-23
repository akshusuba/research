"""
Knowledge Graph Embedding (KGE) baseline models.

Implements:
- TransE: Translation-based embeddings (Bordes et al., 2013)
- DistMult: Bilinear diagonal model (Yang et al., 2015)
- RotatE: Rotation in complex space (Sun et al., 2019)
- ComplEx: Complex-valued embeddings (Trouillon et al., 2016)

All models are adapted for heterogeneous knowledge graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple, Union
import math


class KGEBase(nn.Module):
    """Base class for KGE models."""

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 64,
        margin: float = 1.0,
    ):
        """
        Args:
            num_nodes_dict: Dict mapping node type -> number of nodes
            num_relations: Number of relation types
            embedding_dim: Dimension of embeddings
            margin: Margin for margin-based loss
        """
        super().__init__()
        self.num_nodes_dict = num_nodes_dict
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        # Create separate embeddings for each node type
        self.node_embeddings = nn.ModuleDict()
        for node_type, num_nodes in num_nodes_dict.items():
            self.node_embeddings[node_type] = nn.Embedding(num_nodes, embedding_dim)

        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings."""
        for emb in self.node_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def get_node_embedding(
        self,
        node_type: str,
        node_idx: torch.Tensor
    ) -> torch.Tensor:
        """Get embeddings for nodes of a specific type."""
        return self.node_embeddings[node_type](node_idx)

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute score for triples. To be implemented by subclasses."""
        raise NotImplementedError

    def forward(
        self,
        head_type: str,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_type: str,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores for a batch of triples.

        Args:
            head_type: Type of head nodes
            head_idx: Indices of head nodes
            relation_idx: Indices of relations
            tail_type: Type of tail nodes
            tail_idx: Indices of tail nodes

        Returns:
            Scores for each triple
        """
        head_emb = self.get_node_embedding(head_type, head_idx)
        tail_emb = self.get_node_embedding(tail_type, tail_idx)
        rel_emb = self.relation_embeddings(relation_idx)

        return self.score(head_emb, rel_emb, tail_emb)


class TransEModel(KGEBase):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data.

    Score function: -||h + r - t||
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 64,
        margin: float = 1.0,
        p_norm: int = 2,
    ):
        super().__init__(num_nodes_dict, num_relations, embedding_dim, margin)
        self.p_norm = p_norm

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """TransE scoring: -||h + r - t||_p"""
        return -torch.norm(head + relation - tail, p=self.p_norm, dim=-1)

    def loss(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-based ranking loss."""
        return F.relu(self.margin - pos_score + neg_score).mean()


class DistMultModel(KGEBase):
    """
    DistMult: Embedding Entities and Relations for Learning and
    Inference in Knowledge Bases.

    Score function: <h, r, t> (trilinear dot product)
    """

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """DistMult scoring: sum(h * r * t)"""
        return (head * relation * tail).sum(dim=-1)

    def loss(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss."""
        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()
        return pos_loss + neg_loss


class RotatEModel(KGEBase):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

    Treats embeddings as complex numbers and relations as rotations.
    Score function: -||h ∘ r - t||
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 64,
        margin: float = 1.0,
    ):
        # RotatE uses complex embeddings, so actual dim is 2x
        super().__init__(num_nodes_dict, num_relations, embedding_dim * 2, margin)
        self.complex_dim = embedding_dim

        # Re-initialize relation embeddings for phase
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.uniform_(
            self.relation_embeddings.weight,
            -math.pi,
            math.pi
        )

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """RotatE scoring in complex space."""
        # Split into real and imaginary parts
        head_re, head_im = head.chunk(2, dim=-1)
        tail_re, tail_im = tail.chunk(2, dim=-1)

        # Relation as rotation (phase)
        phase = relation
        rel_re = torch.cos(phase)
        rel_im = torch.sin(phase)

        # Complex multiplication: h ∘ r
        rot_re = head_re * rel_re - head_im * rel_im
        rot_im = head_re * rel_im + head_im * rel_re

        # Distance to tail
        diff_re = rot_re - tail_re
        diff_im = rot_im - tail_im

        # L2 norm
        score = -torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-10).sum(dim=-1)
        return score

    def loss(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
    ) -> torch.Tensor:
        """Self-adversarial negative sampling loss."""
        return F.relu(self.margin - pos_score + neg_score).mean()


class ComplExModel(KGEBase):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction.

    Uses complex-valued embeddings for asymmetric relation modeling.
    Score function: Re(<h, r, conj(t)>)
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 64,
        margin: float = 1.0,
    ):
        # ComplEx uses complex embeddings
        super().__init__(num_nodes_dict, num_relations, embedding_dim * 2, margin)

        # Re-initialize relation embeddings for complex space
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim * 2)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """ComplEx scoring: Re(<h, r, conj(t)>)"""
        head_re, head_im = head.chunk(2, dim=-1)
        rel_re, rel_im = relation.chunk(2, dim=-1)
        tail_re, tail_im = tail.chunk(2, dim=-1)

        # Hermitian dot product
        score = (
            (head_re * rel_re * tail_re).sum(dim=-1) +
            (head_re * rel_im * tail_im).sum(dim=-1) +
            (head_im * rel_re * tail_im).sum(dim=-1) -
            (head_im * rel_im * tail_re).sum(dim=-1)
        )
        return score

    def loss(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss."""
        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()
        return pos_loss + neg_loss


class KGELinkPredictor(nn.Module):
    """
    Wrapper for using KGE models on heterogeneous graphs.

    Handles the conversion between HeteroData format and KGE triple format.
    """

    def __init__(
        self,
        data: HeteroData,
        model_class: type,
        embedding_dim: int = 64,
        **model_kwargs,
    ):
        """
        Args:
            data: PyG HeteroData object
            model_class: KGE model class (TransEModel, DistMultModel, etc.)
            embedding_dim: Embedding dimension
            **model_kwargs: Additional arguments for model
        """
        super().__init__()

        # Get node counts per type
        num_nodes_dict = {}
        for node_type in data.node_types:
            if hasattr(data[node_type], 'num_nodes'):
                num_nodes_dict[node_type] = data[node_type].num_nodes
            elif hasattr(data[node_type], 'x'):
                num_nodes_dict[node_type] = data[node_type].x.size(0)
            else:
                # Try to infer from edges
                max_idx = 0
                for edge_type in data.edge_types:
                    if edge_type[0] == node_type:
                        max_idx = max(max_idx, data[edge_type].edge_index[0].max().item() + 1)
                    if edge_type[2] == node_type:
                        max_idx = max(max_idx, data[edge_type].edge_index[1].max().item() + 1)
                num_nodes_dict[node_type] = max_idx

        # Create relation type to index mapping
        self.edge_type_to_idx = {
            edge_type: i for i, edge_type in enumerate(data.edge_types)
        }
        num_relations = len(data.edge_types)

        # Create KGE model
        self.model = model_class(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            **model_kwargs,
        )

    def forward(
        self,
        edge_type: Tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores for edges.

        Args:
            edge_type: (src_type, relation, dst_type)
            edge_index: [2, num_edges] tensor

        Returns:
            Scores for each edge
        """
        src_type, rel, dst_type = edge_type
        head_idx = edge_index[0]
        tail_idx = edge_index[1]

        rel_idx = torch.full(
            (edge_index.size(1),),
            self.edge_type_to_idx[edge_type],
            dtype=torch.long,
            device=edge_index.device,
        )

        return self.model(src_type, head_idx, rel_idx, dst_type, tail_idx)

    def loss(
        self,
        edge_type: Tuple[str, str, str],
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for positive and negative edges."""
        pos_scores = self.forward(edge_type, pos_edge_index)
        neg_scores = self.forward(edge_type, neg_edge_index)
        return self.model.loss(pos_scores, neg_scores)

    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get node embeddings for all types."""
        return {
            node_type: emb.weight.detach()
            for node_type, emb in self.model.node_embeddings.items()
        }
