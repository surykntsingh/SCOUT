import torch
import torch.nn as nn

from modules.common import SublayerConnection, LayerNorm
from utils import utils
from utils.utils import clones

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward,gate_fusion, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attns = src_attn
        # self.feed_forward = feed_forward
        self.gate_fusion = gate_fusion
        self.feed_forward = feed_forward
        # self.ff_2 = ff_2
        self.n = 6
        self.sublayers = clones(SublayerConnection(d_model, dropout), self.n)


    def forward(self, x, patch_features, slide_features, concept_features, src_mask, tgt_mask):
        # x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))[0]

        # 1. Self-attention (causal)
        x = self.sublayers[0](
            x,
            lambda x: self.self_attn(x, x, x, tgt_mask)[0]
        )

        # 2. Cross-attention: patch
        x_patch = self.sublayers[1](
            x,
            lambda x: self.src_attns[0](x, patch_features, patch_features, src_mask)[0]
        )

        # 3. Cross-attention: slide
        x_slide = self.sublayers[2](
            x,
            lambda x: self.src_attns[1](x, slide_features, slide_features, src_mask)[0]
        )

        # 4. Cross-attention: concept
        x_concept = self.sublayers[3](
            x,
            lambda x: self.src_attns[2](x, concept_features, concept_features, src_mask)[0]
        )

        # 5. Gated multimodal fusion (residual inside)
        x, weights = self.sublayers[4](
            x,
            lambda x: self.gate_fusion(x, x_patch, x_slide, x_concept)
        )

        # 6. Feed-forward
        x = self.sublayers[5](x, self.feed_forward)

        return x, weights

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, patch_features, slide_features, concept_features, src_mask, tgt_mask):
        attn_maps = []
        # attn_img_all = []
        # attn_con_all = []
        for layer in self.layers:
            x, alpha = layer(x, patch_features, slide_features, concept_features, src_mask, tgt_mask)
            attn_maps.append(alpha)
            # attn_img_all.append(attn_img)
            # attn_con_all.append(attn_con)


        # attn_img_all = torch.stack(attn_img_all)  # (layers, batch, heads, seq_len, src_len)
        # attn_con_all = torch.stack(attn_con_all)
        attn_maps = torch.stack(attn_maps).mean(0)
        return self.norm(x), attn_maps #(attn_maps, attn_img_all, attn_con_all)