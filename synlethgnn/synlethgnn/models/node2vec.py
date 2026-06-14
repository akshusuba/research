"""node2vec / DeepWalk baseline -- structure via memorized embeddings.

We implement random-walk + skip-gram embeddings from scratch (no pyg-lib
dependency) so the baseline runs anywhere. It learns one embedding per node
from uniform random walks over the *training* graph, then a small decoder
scores pairs. It exploits graph structure, so it is a strong competitor in the
transductive regime. But its embeddings are look-up tables: a gene held out of
the training graph (inductive/cold-gene setting) never receives a meaningful
embedding, so this baseline is expected to collapse there. That contrast --
strong transductive, weak inductive -- is exactly what motivates an inductive
GNN.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


def _build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> List[np.ndarray]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    ei = edge_index.cpu().numpy()
    for s, d in zip(ei[0], ei[1]):
        adj[s].append(d)
    return [np.array(a, dtype=np.int64) for a in adj]


def _generate_walks(adj, num_nodes, walk_length, walks_per_node, rng):
    walks = []
    nodes = np.arange(num_nodes)
    for _ in range(walks_per_node):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [int(start)]
            cur = start
            for _ in range(walk_length - 1):
                nbrs = adj[cur]
                if len(nbrs) == 0:
                    break
                cur = int(nbrs[rng.integers(len(nbrs))])
                walk.append(cur)
            walks.append(walk)
    return walks


def _skipgram_pairs(walks, context_size):
    centers, contexts = [], []
    for walk in walks:
        L = len(walk)
        for i in range(L):
            lo = max(0, i - context_size)
            hi = min(L, i + context_size + 1)
            for j in range(lo, hi):
                if j != i:
                    centers.append(walk[i])
                    contexts.append(walk[j])
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


class Node2VecLinkPredictor(nn.Module):
    requires_embedding_fit = True

    def __init__(self, in_channels, model_cfg, num_nodes: int,
                 embedding_dim: int | None = None, walk_length: int = 15,
                 context_size: int = 5, walks_per_node: int = 10,
                 n2v_epochs: int = 5, n2v_lr: float = 0.01,
                 neg_samples: int = 5):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim or model_cfg.out_channels
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.n2v_epochs = n2v_epochs
        self.n2v_lr = n2v_lr
        self.neg_samples = neg_samples
        self.register_buffer("emb", torch.zeros(num_nodes, self.embedding_dim))
        self.decoder = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, model_cfg.hidden_channels), nn.ReLU(),
            nn.Linear(model_cfg.hidden_channels, 1),
        )

    def fit_embeddings(self, edge_index, device="cpu"):
        if edge_index.size(1) == 0:
            return
        rng = np.random.default_rng(0)
        adj = _build_adjacency(edge_index, self.num_nodes)
        walks = _generate_walks(adj, self.num_nodes, self.walk_length,
                                self.walks_per_node, rng)
        centers, contexts = _skipgram_pairs(walks, self.context_size)
        if len(centers) == 0:
            return

        dev = torch.device(device)
        emb_in = nn.Embedding(self.num_nodes, self.embedding_dim).to(dev)
        emb_out = nn.Embedding(self.num_nodes, self.embedding_dim).to(dev)
        nn.init.normal_(emb_in.weight, std=0.1)
        nn.init.normal_(emb_out.weight, std=0.1)
        opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()),
                               lr=self.n2v_lr)

        centers_t = torch.from_numpy(centers).to(dev)
        contexts_t = torch.from_numpy(contexts).to(dev)
        n = centers_t.size(0)
        batch = 4096
        for _ in range(self.n2v_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                c = centers_t[idx]
                pos = contexts_t[idx]
                neg = torch.randint(0, self.num_nodes, (idx.size(0), self.neg_samples),
                                    device=dev)
                v_c = emb_in(c)                       # (B, D)
                v_pos = emb_out(pos)                  # (B, D)
                v_neg = emb_out(neg)                  # (B, K, D)
                pos_score = (v_c * v_pos).sum(-1)
                neg_score = torch.bmm(v_neg, v_c.unsqueeze(-1)).squeeze(-1)
                loss = -(torch.log(torch.sigmoid(pos_score) + 1e-9).mean()
                         + torch.log(torch.sigmoid(-neg_score) + 1e-9).mean())
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            self.emb.copy_(emb_in.weight.data.to(self.emb.device))

    def encode(self, x, edge_index=None, edge_type=None):
        return self.emb

    def decode(self, z, pairs):
        zi, zj = z[pairs[0]], z[pairs[1]]
        feat = torch.cat([zi + zj, (zi - zj).abs()], dim=-1)
        return self.decoder(feat).squeeze(-1)

    def forward(self, x, edge_index, pairs, edge_type=None):
        return self.decode(self.emb, pairs)
