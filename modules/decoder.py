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
        self.n = 5
        self.sublayer = clones(SublayerConnection(d_model, dropout), self.n)


    def forward(self, feats, hidden_states, concepts, src_mask, tgt_mask):
        # m = hidden_states
        x0 = self.sublayer[0](feats, lambda x: self.self_attn(x, x, x, tgt_mask))

        # print(f'**************x: {feats.shape},  hidden_states: {hidden_states.shape}. x0: {x0.shape}')

        x1 = self.sublayer[1](x0, lambda x: self.src_attn(x, hidden_states, hidden_states, src_mask))
        # print(f'x1: {x1.shape}, concepts: {concepts.shape}')
        x2 = self.sublayer[2](x0, lambda x: self.concept_attn(x, concepts, concepts))
        # print(f'---->>x2: {x2.shape}, concepts: {concepts.shape}')
        x = self.sublayer[3](x1, self.feed_forward)+self.sublayer[4](x2, self.feed_forward)
        # print(f'---->>x: {x.shape}, concepts: {concepts.shape}')
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, concepts, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, hidden_states, concepts, src_mask, tgt_mask)
        return self.norm(x)