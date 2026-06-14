"""The GNN link predictor -- the model that should *earn its keep*.

An encoder propagates information over the gene interaction graph so that each
gene's embedding reflects its neighborhood (its pathway/module context). A
symmetric decoder then scores a gene pair. Because synthetic lethality arises
from the *relationship* between two genes' network positions, this is exactly
the inductive bias the task rewards.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, FastRGCNConv


def _make_conv(name: str, in_dim: int, out_dim: int, num_relations: int = 1):
    if name == "sage":
        return SAGEConv(in_dim, out_dim)
    if name == "gcn":
        return GCNConv(in_dim, out_dim)
    if name == "gat":
        # 4 heads, concatenated -> out_dim must be divisible by heads
        heads = 4
        assert out_dim % heads == 0, "out_channels must be divisible by 4 for GAT"
        return GATConv(in_dim, out_dim // heads, heads=heads)
    if name == "rgcn":
        # FastRGCNConv is markedly faster on CPU; basis decomposition keeps
        # params manageable across several relations.
        num_bases = min(num_relations, 4)
        return FastRGCNConv(in_dim, out_dim, num_relations=num_relations,
                            num_bases=num_bases)
    raise ValueError(f"Unknown encoder: {name}")


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.3, encoder="sage", num_relations=1):
        super().__init__()
        self.dropout = dropout
        self.is_relational = (encoder == "rgcn")
        self.convs = nn.ModuleList()
        mk = lambda i, o: _make_conv(encoder, i, o, num_relations)
        if num_layers == 1:
            self.convs.append(mk(in_channels, out_channels))
        else:
            self.convs.append(mk(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(mk(hidden_channels, hidden_channels))
            self.convs.append(mk(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_type=None):
        for i, conv in enumerate(self.convs):
            if self.is_relational:
                x = conv(x, edge_index, edge_type)
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BilinearDecoder(nn.Module):
    """Symmetric bilinear score; symmetry matters because SL is undirected."""

    def __init__(self, dim):
        super().__init__()
        self.rel = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.rel)

    def forward(self, z, pairs):
        zi = z[pairs[0]]
        zj = z[pairs[1]]
        rel_sym = 0.5 * (self.rel + self.rel.t())
        return (zi @ rel_sym * zj).sum(dim=-1)


class DotDecoder(nn.Module):
    def forward(self, z, pairs):
        return (z[pairs[0]] * z[pairs[1]]).sum(dim=-1)


class MLPDecoder(nn.Module):
    """Symmetric MLP decoder over the (sum, abs-diff) of the two embeddings."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, pairs):
        zi, zj = z[pairs[0]], z[pairs[1]]
        feat = torch.cat([zi + zj, (zi - zj).abs()], dim=-1)
        return self.net(feat).squeeze(-1)


def _make_decoder(name: str, dim: int):
    if name == "bilinear":
        return BilinearDecoder(dim)
    if name == "dot":
        return DotDecoder()
    if name == "mlp":
        return MLPDecoder(dim)
    raise ValueError(f"Unknown decoder: {name}")


class GNNLinkPredictor(nn.Module):
    """Encoder + decoder bundle exposing a uniform link-prediction API."""

    def __init__(self, in_channels, model_cfg, num_relations=1):
        super().__init__()
        self.encoder = GNNEncoder(
            in_channels, model_cfg.hidden_channels, model_cfg.out_channels,
            num_layers=model_cfg.num_layers, dropout=model_cfg.dropout,
            encoder=model_cfg.encoder, num_relations=num_relations,
        )
        self.decoder = _make_decoder(model_cfg.decoder, model_cfg.out_channels)

    def encode(self, x, edge_index, edge_type=None):
        return self.encoder(x, edge_index, edge_type)

    def decode(self, z, pairs):
        return self.decoder(z, pairs)

    def forward(self, x, edge_index, pairs, edge_type=None):
        z = self.encode(x, edge_index, edge_type)
        return self.decode(z, pairs)
