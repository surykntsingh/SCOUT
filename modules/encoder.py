import torch.nn as nn
import torch
import torch.nn.functional as F

from modules.common import LayerNorm, SublayerConnection
from utils import utils
from utils.utils import clones


class Encoder(nn.Module):
    def __init__(self, layer, N, PAM, concept_fusion):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.PAM = clones(PAM, N)
        self.N = N
        self.concept_fusion = concept_fusion
        self.layer_weights = nn.Parameter(torch.ones(N))

    def forward(self, x, mask, concepts):
        s=[]
        for i,layer in enumerate(self.layers):
            x = self.concept_fusion(x, concepts)
            x = layer(x, mask)

            s.append(self.PAM[i](x))


        # o = s[0]
        # for i in range(1,len(s)):
        #     o +=  s[i]
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
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)