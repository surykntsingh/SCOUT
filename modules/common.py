import numpy as np
import torch
import torch.nn as nn

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(self.norm(x))
        if type(y)==tuple:
            return x + self.dropout(y[0]), y[1]
        else:
            return x + self.dropout(y)

class ConceptSublayer(nn.Module):
    def __init__(self, d_model, concept_fusion):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fusion = concept_fusion   # expects fused output, same shape as x

    def forward(self, x, concepts):
        x_infused, _, _ = self.fusion(self.norm(x), concepts)
        return x + x_infused