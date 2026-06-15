"""Spatial GNN graph classifier -- reads cell ARRANGEMENT.

Message passing over the spatial cell graph followed by a global readout, so the
patient-level prediction depends on how cells are organized (e.g., tumor-immune
contacts), not just which cells are present. This is the only model in the suite
that can see arrangement; the composition baselines cannot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_max_pool


def _make_conv(name, in_dim, out_dim):
    if name == "sage":
        return SAGEConv(in_dim, out_dim)
    if name == "gcn":
        return GCNConv(in_dim, out_dim)
    if name == "gin":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        return GINConv(mlp)
    raise ValueError(f"Unknown encoder: {name}")


class GNNGraphClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, model_cfg):
        super().__init__()
        self.dropout = model_cfg.dropout
        self.pool = global_max_pool if model_cfg.pool == "max" else global_mean_pool
        h = model_cfg.hidden_channels
        self.convs = nn.ModuleList()
        self.convs.append(_make_conv(model_cfg.encoder, in_channels, h))
        for _ in range(model_cfg.num_layers - 1):
            self.convs.append(_make_conv(model_cfg.encoder, h, h))
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(model_cfg.dropout),
            nn.Linear(h, num_classes))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = self.pool(x, batch)
        return self.head(g)
