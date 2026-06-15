"""GIN graph classifier for binary molecular activity prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool


class GIN(nn.Module):
    def __init__(self, in_dim, hidden=128, num_layers=4, dropout=0.3, pool="mean"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden
            mlp = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.pool = global_mean_pool if pool == "mean" else global_add_pool
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        return self.head(x).squeeze(-1)
