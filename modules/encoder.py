from pyexpat import features

import torch.nn as nn
import torch
import torch.nn.functional as F

from modules.common import LayerNorm, SublayerConnection, ConceptSublayer
from utils import utils
from utils.utils import clones

class FilmFusion(nn.Module):
    def __init__(self, D, D_s, hidden=1024, dropout=0.4, mod_alpha=0.5):
        super().__init__()
        self.gamma_beta = nn.Sequential(
            nn.Linear(D_s, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2*D),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2*D, 2 * D)     # gamma, beta
        )
        self.layernorm = nn.LayerNorm(D)
        self.mod_alpha = mod_alpha

    def forward(self, patch, slide):
        # patch: [B,M,D], slide:[B,D_s]
        gb = self.gamma_beta(slide)  # [B, 2D]
        gamma, beta = gb.chunk(2, dim=-1)  # [B,D], [B,D]
        gamma = gamma.unsqueeze(1)  # [B,1,D]
        beta = beta.unsqueeze(1)
        out = self.layernorm(patch * (1 + self.mod_alpha * gamma) + beta)
        return out  # [B,M,D]

class Encoder(nn.Module):
    def __init__(self, layer, N, PAM):
        super().__init__()
        self.layers = clones(layer, N)
        self.patch_norm = LayerNorm(layer.d_model)
        self.slide_norm = LayerNorm(layer.d_model)
        self.concept_norm = LayerNorm(layer.d_model)
        self.norm = LayerNorm(layer.d_model)

        self.PAM = clones(PAM, N)
        self.N = N
        # self.concept_fusion = concept_fusion
        slide_fusions = FilmFusion(768,768)
        concept_fusions = FilmFusion(768, 768)
        self.slide_fusion_layer = clones(slide_fusions, self.N)
        self.concept_fusion_layer = clones(concept_fusions, self.N)
        self.patch_layer_weights = nn.Parameter(torch.ones(N))
        self.slide_layer_weights = nn.Parameter(torch.ones(N))
        self.concept_layer_weights = nn.Parameter(torch.ones(N))



    def forward(self, patch, slide, concept, mask):
        patches = []
        slides = []
        concepts = []

        x_patch = patch

        for i in range(self.N):

            x_patch = self.layers[i](self.norm(x_patch), mask)
            x_patch = self.PAM[i](x_patch)
            if not x_slide:
                x_slide = x_patch
            x_slide = self.slide_fusion_layer[i](x_slide, slide)

            if not x_concepts:
                x_concepts = x_patch
            x_concept = self.concept_fusion_layer[i](x_concept, concept)

            patches.append(x_patch)
            slides.append(x_slide)
            concepts.append(x_concept)

        features = [
            self.aggregate_weights(patches, self.patch_layer_weights, self.patch_norm),
            self.aggregate_weights(slides, self.slide_layer_weights, self.slide_norm),
            self.aggregate_weights(concepts, self.concept_layer_weights, self.concept_norm)
        ]

        return features

    def aggregate_weights(self, s, layer_weights, norm):
        s = torch.stack(s, dim=0)  # [N, B, L, D]
        w = F.softmax(layer_weights, dim=0)  # [N]
        o = (w[:, None, None, None] * s).sum(0)  # [B, L, D]

        return norm(o)




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