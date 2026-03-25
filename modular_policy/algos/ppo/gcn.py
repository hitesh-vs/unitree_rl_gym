import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modular_policy.config import cfg


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        self.linear       = nn.Linear(in_dim, out_dim)
        self.norm         = nn.LayerNorm(out_dim)
        self.use_activation = activation

    def forward(self, X, A_norm):
        AX  = (A_norm @ X).detach()
        out = self.linear(AX)
        out = self.norm(out)
        if self.use_activation:
            out = F.relu(out)
        return out


class MorphologyGCN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=16, out_dim=8, num_layers=2):
        super().__init__()
        self.out_dim = out_dim
        layers = []
        in_dim = node_feat_dim
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(in_dim, hidden_dim, activation=True))
            in_dim = hidden_dim
        layers.append(GCNLayer(in_dim, out_dim, activation=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, X, A_norm):
        h = X
        for layer in self.layers:
            h = layer(h, A_norm)
        return h


def build_gcn_from_cfg(cfg):
    mode = cfg.MODEL.GRAPH_ENCODING
    if mode == "none":
        return None
    feat_dim = {
        "onehot":      7,
        "topological": 6,
        "rwse":        cfg.MODEL.RWSE_K,
    }[mode]
    return MorphologyGCN(
        node_feat_dim=feat_dim,
        hidden_dim=cfg.MODEL.GCN.HIDDEN_DIM,
        out_dim=cfg.MODEL.GCN.OUT_DIM,
        num_layers=cfg.MODEL.GCN.NUM_LAYERS,
    )