"""Feature-only MLP baseline -- the structure-blind control.

This model never sees ``edge_index``. It encodes each gene from its node
features alone and scores a pair with a symmetric MLP decoder. It is the most
important baseline in the project: if synthetic lethality is genuinely a
topological property, this model *cannot* solve the task, and the gap between
it and the GNN is precisely the value the graph adds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLinkPredictor(nn.Module):
    def __init__(self, in_channels, model_cfg):
        super().__init__()
        self.dropout = model_cfg.dropout
        h = model_cfg.hidden_channels
        out = model_cfg.out_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, h), nn.ReLU(), nn.Dropout(model_cfg.dropout),
            nn.Linear(h, out),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * out, h), nn.ReLU(),
            nn.Linear(h, 1),
        )

    def encode(self, x, edge_index=None, edge_type=None):
        # edge_index/edge_type accepted for API compatibility, intentionally ignored.
        return self.encoder(x)

    def decode(self, z, pairs):
        zi, zj = z[pairs[0]], z[pairs[1]]
        feat = torch.cat([zi + zj, (zi - zj).abs()], dim=-1)
        return self.decoder(feat).squeeze(-1)

    def forward(self, x, edge_index, pairs, edge_type=None):
        z = self.encode(x)
        return self.decode(z, pairs)
