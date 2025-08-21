import torch
import torch.nn as nn

from modules.common import SublayerConnection, LayerNorm
from utils import utils
from utils.utils import clones


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, concept_attn, feed_forward, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.concept_attn = concept_attn
        self.feed_forward = feed_forward
        self.n = 4
        self.sublayer = clones(SublayerConnection(d_model, dropout), self.n)


    def forward(self, x, hidden_states, concepts, src_mask, tgt_mask):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # print(f'x: {x.shape}')

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        print(f'x: {x.shape}')
        x = self.sublayer[2](x, lambda x: self.concept_attn(x, concepts, concepts))
        return self.sublayer[3](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, concepts, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, hidden_states, concepts, src_mask, tgt_mask)
        return self.norm(x)