from pyexpat import features

import torch.nn as nn
import torch
import torch.nn.functional as F

from modules.common import LayerNorm, SublayerConnection, ConceptSublayer
from utils import utils
from utils.utils import clones

class FilmFusion(nn.Module):
    def __init__(self, D, D_s, hidden=1024, dropout=0.4):
        super().__init__()
        self.gamma_beta = nn.Sequential(
            nn.Linear(D_s, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(D, 2 * D)     # gamma, beta
        )
        self.layernorm = nn.LayerNorm(D)

    def forward(self, patch, slide):
        # patch: [B,M,D], slide:[B,D_s]
        gb = self.gamma_beta(slide)  # [B, 2D]
        gamma, beta = gb.chunk(2, dim=-1)  # [B,D], [B,D]
        gamma = gamma.unsqueeze(1)  # [B,1,D]
        beta = beta.unsqueeze(1)
        out = self.layernorm(patch * (1 + gamma) + beta)
        return out  # [B,M,D]

class ChannelProjector(nn.Module):
    def __init__(self, n_concepts, d_model, dropout, hidden=256):
        super().__init__()
        # Option A: per-concept learned embedding table (concept id -> vector)
        self.id_embed = nn.Embedding(n_concepts, d_model)
        # Projection from scalar activation (score) to scale per concept
        self.score_proj = nn.Sequential(
            nn.Linear(1, hidden),
            # nn.ReLU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
            # nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, scores):
        # scores: [B, M] (float activations from GECKO)
        B, M = scores.shape
        ids = torch.arange(M, device=scores.device).unsqueeze(0).expand(B, M)  # [B, M]
        base = self.id_embed(ids)  # [B, M, d_model]
        # project scalar score per concept to a vector and use as multiplicative gating
        scales = self.score_proj(scores.unsqueeze(-1))  # [B, M, d_model]
        concept_tokens = base * (1 + scales)  # broadcast multiply
        concept_tokens = self.layernorm(concept_tokens)
        return concept_tokens  # [B, M, d_model]

class Encoder(nn.Module):
    def __init__(self, layer, N, PAM, concept_fusion):
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
        for i,layer in enumerate(self.layers):

            x_patch = layer(self.norm(x_patch), mask)
            x_patch = self.PAM[i](x_patch)
            x_slide = self.slide_fusion_layer[i](x_patch, slide)
            x_concept = self.concept_fusion_layer[i](x_patch, concept)

            patches.append(x_patch)
            slides.append(x_slide)
            concepts.append(x_concept)

        features = torch.stack([
            self.aggregate_weights(patches, self.patch_layer_weights, self.patch_norm),
            self.aggregate_weights(slides, self.slide_layer_weights, self.slide_norm),
            self.aggregate_weights(concepts, self.concept_layer_weights, self.concept_norm)
        ], dim =-1)

        return features

    def aggregate_weights(self, s, layer_weights, norm):
        s = torch.stack(s, dim=0)  # [N, B, L, D]
        w = F.softmax(layer_weights, dim=0)  # [N]
        o = (w[:, None, None, None] * s).sum(0)  # [B, L, D]

        return norm(o).unsqueeze(-1)




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