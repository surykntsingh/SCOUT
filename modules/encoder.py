import torch.nn as nn
import torch
import torch.nn.functional as F

from modules.common import LayerNorm, SublayerConnection, ConceptSublayer
from utils import utils
from utils.utils import clones


class Encoder(nn.Module):
    def __init__(self, layer, N, PAM, concept_fusion):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.PAM = clones(PAM, N)
        self.N = N
        self.concept_sublayers = clones(ConceptSublayer(layer.d_model, concept_fusion), N)
        self.layer_weights = nn.Parameter(torch.ones(N))

    def forward(self, x, mask, concepts):
        s=[]
        for i,layer in enumerate(self.layers):
            # x = self.concept_fusion(self.norm(x), concepts)
            x = layer(self.norm(x), mask)
            # x = self.concept_sublayers[i](x, concepts)
            # x = self.PAM[i](x)

            s.append(x)

            # Weighted sum of layer outputs
        s = torch.stack(s, dim=0)  # [N, B, L, D]
        w = F.softmax(self.layer_weights, dim=0)  # [N]
        o = (w[:, None, None, None] * s).sum(0)  # [B, L, D]

        return self.norm(o)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)[0])
        return self.sublayer[1](x, self.feed_forward)