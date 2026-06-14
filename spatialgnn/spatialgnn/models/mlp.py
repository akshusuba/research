"""MLP node classifier -- the structure-blind control.

Classifies each cell from its own expression vector, with no access to the
spatial graph. If domain identity is genuinely a neighbourhood property, this
model is bounded by how much a single (noisy) cell reveals -- and the gap to
the GNN is the value the spatial graph adds.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class MLPNodeClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, model_cfg):
        super().__init__()
        self.dropout = model_cfg.dropout
        h = model_cfg.hidden_channels
        layers = [nn.Linear(in_channels, h), nn.ReLU(), nn.Dropout(model_cfg.dropout)]
        for _ in range(max(0, model_cfg.num_layers - 1)):
            layers += [nn.Linear(h, h), nn.ReLU(), nn.Dropout(model_cfg.dropout)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(h, num_classes)

    def forward(self, x, edge_index=None):
        # edge_index accepted for a uniform API and intentionally ignored.
        return self.head(self.body(x))
