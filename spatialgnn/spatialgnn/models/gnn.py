"""Spatial GNN node classifier -- the model that should earn its keep.

It propagates expression over the spatial neighbourhood graph, so each cell's
representation pools information from the cells physically around it. Because a
cell's domain is a property of its location (shared with its neighbours), this
neighbourhood aggregation is exactly the inductive bias the task rewards.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


def _make_conv(name, in_dim, out_dim):
    if name == "sage":
        return SAGEConv(in_dim, out_dim)
    if name == "gcn":
        return GCNConv(in_dim, out_dim)
    if name == "gat":
        heads = 4
        assert out_dim % heads == 0
        return GATConv(in_dim, out_dim // heads, heads=heads)
    raise ValueError(f"Unknown encoder: {name}")


class GNNNodeClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, model_cfg):
        super().__init__()
        self.dropout = model_cfg.dropout
        h = model_cfg.hidden_channels
        L = model_cfg.num_layers
        self.convs = nn.ModuleList()
        if L == 1:
            self.convs.append(_make_conv(model_cfg.encoder, in_channels, h))
        else:
            self.convs.append(_make_conv(model_cfg.encoder, in_channels, h))
            for _ in range(L - 1):
                self.convs.append(_make_conv(model_cfg.encoder, h, h))
        self.head = nn.Linear(h, num_classes)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)
