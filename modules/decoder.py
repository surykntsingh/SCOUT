import torch
import torch.nn as nn

from modules.common import SublayerConnection, LayerNorm
from utils import utils
from utils.utils import clones

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, concept_attn, ff_1,gate_fusion, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.concept_attn = concept_attn
        # self.feed_forward = feed_forward
        self.gate_fusion = gate_fusion
        self.ff_1 = ff_1
        # self.ff_2 = ff_2
        self.n = 4
        self.sublayer = clones(SublayerConnection(d_model, dropout), self.n)


    def forward(self, feats, hidden_states, concepts, src_mask, tgt_mask):
        # m = hidden_states
        x = self.sublayer[0](feats, lambda x: self.self_attn(x, x, x, tgt_mask))

        # print(f'**************x: {feats.shape},  hidden_states: {hidden_states.shape}. x0: {x0.shape}')

        x_img = self.sublayer[1](x, lambda x: self.src_attn(x, hidden_states, hidden_states, src_mask))
        # print(f'x1: {x1.shape}, concepts: {concepts.shape}')
        x_con = self.sublayer[2](x, lambda x: self.concept_attn(x, concepts, concepts, mask=None))

        # reinforce linguistic grounding
        x_con = x_con + 0.1 * x
        # print(f'---->>x2: {x2.shape}, concepts: {concepts.shape}')
        # x = self.sublayer[3](x1, self.ff_1) + self.sublayer[4](x2, self.ff_2)
        # print(f'---->>x: {x.shape}, concepts: {concepts.shape}')
        x_fused, alpha = self.gate_fusion(x, x_img, x_con)
        out = self.sublayer[3](x_fused, self.ff_1)
        return out, alpha

class Decoder(nn.Module):
    def __init__(self, layer, feed_forward, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, concepts, src_mask, tgt_mask):
        attn_maps = []
        for layer in self.layers:
            x, alpha = layer(x, hidden_states, concepts, src_mask, tgt_mask)
            attn_maps.append(alpha)

        return self.norm(x), attn_maps