import torch
import torch.nn as nn

from modules.common import SublayerConnection, LayerNorm
from utils import utils
from utils.utils import clones

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, ff_1,gate_fusion, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        # self.feed_forward = feed_forward
        self.gate_fusion = gate_fusion
        self.ff_1 = ff_1
        # self.ff_2 = ff_2
        self.n = 5
        self.sublayer = clones(SublayerConnection(d_model, dropout), self.n)


    def forward(self, feats, hidden_states, src_mask, tgt_mask):
        # m = hidden_states
        x_self, _ = self.sublayer[0](feats, lambda x: self.self_attn(x, x, x, tgt_mask))

        # print(f'hidden_states: {hidden_states.shape}')
        patch_features = hidden_states[...,0]
        slide_features = hidden_states[...,1]
        concept_features = hidden_states[...,2]
        # print(f'patch_features: {patch_features.shape}, src_mask: {src_mask.shape}')

        x_patch, _ = self.sublayer[1](x_self, lambda x: self.src_attn[0](x, patch_features, patch_features, src_mask))
        x_slide, _ = self.sublayer[2](x_self, lambda x: self.src_attn[1](x, slide_features, slide_features, src_mask))
        x_concept, _ = self.sublayer[3](x_self, lambda x: self.src_attn[2](x, concept_features, concept_features, src_mask))

        x_fused, alpha = self.gate_fusion(x_self, x_patch, x_slide, x_concept)

        out = self.sublayer[4](x_fused, self.ff_1)
        return out, alpha

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        attn_maps = []
        # attn_img_all = []
        # attn_con_all = []
        for layer in self.layers:
            x, alpha = layer(x, hidden_states, src_mask, tgt_mask)
            attn_maps.append(alpha)
            # attn_img_all.append(attn_img)
            # attn_con_all.append(attn_con)


        # attn_img_all = torch.stack(attn_img_all)  # (layers, batch, heads, seq_len, src_len)
        # attn_con_all = torch.stack(attn_con_all)
        return self.norm(x), attn_maps #(attn_maps, attn_img_all, attn_con_all)