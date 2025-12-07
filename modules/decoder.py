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
        x_self, _ = self.sublayer[0](feats, lambda x: self.self_attn(x, x, x, tgt_mask))

        x_img, x_img_attn = self.sublayer[1](x_self, lambda x: self.src_attn(x, hidden_states, hidden_states, src_mask))

        x_con, x_con_attn = self.sublayer[2](x_self, lambda x: self.concept_attn(x, concepts, concepts, mask=None))

        x_fused, gate_weights = self.gate_fusion(feats, x_self, x_img, x_con)

        out = self.sublayer[3](x_fused, self.ff_1)
        return out, (gate_weights, x_img_attn, x_con_attn)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, concepts, src_mask, tgt_mask):
        gate_weights = []
        attn_img_all = []
        attn_con_all = []
        for layer in self.layers:
            x, (weights, attn_img, attn_con) = layer(x, hidden_states, concepts, src_mask, tgt_mask)
            gate_weights.append(weights)
            attn_img_all.append(attn_img)
            attn_con_all.append(attn_con)

        attn_img_all = torch.stack(attn_img_all)  # (layers, batch, heads, seq_len, src_len)
        attn_con_all = torch.stack(attn_con_all)
        # gate_weights = torch.stack(gate_weights).mean(dim=0)

        return self.norm(x), (gate_weights, attn_img_all, attn_con_all)