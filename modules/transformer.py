from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention_model import AttModel
from modules.common import subsequent_mask, LayerNorm
from modules.decoder import DecoderLayer, Decoder
from modules.encoder import Encoder, EncoderLayer
from utils.utils import pad_tokens, pack_wrapper, clones


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed,slide_embed, concept_embed):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.concept_embed = concept_embed
        self.slide_embed = slide_embed


    def forward(self, patch, slide, concepts, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(patch, slide,concepts, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, patch, slide,concepts, src_mask):
        # print(f'src: {src.shape}')
        return self.encoder(self.src_embed(patch),self.slide_embed(slide), self.concept_embed(concepts), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).reshape(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), attn

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        # kt =
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadGatedFusion(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # gating network per head
        self.gate_net_1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.gate_net_2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, x_self, x_img, x_con):
        """
        x: [B, L, D] - decoder token
        x_img: [B, L, D] - attended image features
        x_con: [B, L, D] - attended concept features
        """
        B, L, D = x.shape
        H = self.num_heads
        d_h = self.head_dim

        # compute per-head gating
        alpha = self.gate_net_1(x).view(B, L, H, d_h)   # [B, L, H, d_h]
        beta = self.gate_net_2(x).view(B, L, H, d_h)  # [B, L, H, d_h]

        # reshape inputs to [B, L, H, d_h]
        x_self = x_self.view(B, L, H, d_h)
        x_img = x_img.view(B, L, H, d_h)
        x_con = x_con.view(B, L, H, d_h)

        # fuse per head
        fused = beta * x_img + alpha * x_con + (1-alpha-beta) * x_self

        # reshape back and project
        fused = fused.view(B, L, D)
        out = self.out_proj(fused)  # residual connection + projection
        return out, alpha


class MultiHeadGatedFusionV2(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        H = num_heads
        d_h = d_model // H
        self.num_heads = H
        self.head_dim = d_h

        # produce 3 logits for gating per head
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3 * d_model)   # 3-way gating
        )

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, x_self, x_img, x_con):
        B, L, D = x.shape
        H, d_h = self.num_heads, self.head_dim

        gates = self.gate_net(x).view(B, L, H, 3, d_h)     # [B,L,H,3,d_h]
        weights = gates.softmax(dim=3)                     # [B,L,H,3,d_h]

        w_self = weights[:, :, :, 0]
        w_img  = weights[:, :, :, 1]
        w_con  = weights[:, :, :, 2]

        # reshape inputs
        x_self = x_self.view(B, L, H, d_h)
        x_img  = x_img.view(B, L, H, d_h)
        x_con  = x_con.view(B, L, H, d_h)

        fused = w_self * x_self + w_img * x_img + w_con * x_con

        fused = fused.view(B, L, D)
        return self.out_proj(fused), weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class MultiHeadGatedFusionV3(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, temperature=1.0):
        super().__init__()
        H = num_heads
        d_h = d_model // H

        self.num_heads = H
        self.head_dim = d_h
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Per-head projections
        self.proj_self = nn.Linear(d_model, d_model)
        self.proj_img  = nn.Linear(d_model, d_model)
        self.proj_con  = nn.Linear(d_model, d_model)

        # Contextual gating network
        self.gate_net = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3 * d_model)
        )

        # Fusion + normalization
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Small FFN for extra expressiveness
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, x_patch, x_slide, x_concept):
        B, L, D = x.shape
        H, d_h = self.num_heads, self.head_dim

        # ---- Per-head projections ----
        s0 = self.proj_self(x_patch)
        i0 = self.proj_img(x_slide)
        c0 = self.proj_con(x_concept)

        # ---- Contextual gating ----
        ctx = torch.cat([x, x_slide, x_concept], dim=-1)
        gates = self.gate_net(ctx)                         # [B,L,3D]
        gates = gates.view(B, L, H, 3, d_h)

        # Temperature-scaled softmax
        weights = F.softmax(gates / self.temperature, dim=3)

        w_patch = weights[:, :, :, 0]
        w_slide  = weights[:, :, :, 1]
        w_concept  = weights[:, :, :, 2]

        # ---- Reshape modalities ----
        s0 = s0.view(B, L, H, d_h)
        i0 = i0.view(B, L, H, d_h)
        c0 = c0.view(B, L, H, d_h)

        # ---- Weighted fusion ----
        fused = w_patch * s0 + w_slide * i0 + w_concept * c0
        fused = fused.view(B, L, D)

        # ---- Projection + residual + normalization ----
        out = x + self.out_proj(fused)
        out = self.norm(out)

        # ---- Extra FFN ----
        out = out + self.ffn(out)

        return out, weights

class MultiHeadGatedFusionV4(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2, temperature=1.0):
        super().__init__()
        H = num_heads
        d_h = d_model // H

        self.num_heads = H
        self.head_dim = d_h
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Per-head projections
        self.proj_self = nn.Linear(d_model, d_model)
        self.proj_img  = nn.Linear(d_model, d_model)
        self.proj_con  = nn.Linear(d_model, d_model)

        # Contextual gating network
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, 4*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, 2*d_model)
        )

        # Fusion + normalization
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Small FFN for extra expressiveness
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x_self, x_img, x_con):
        B, L, D = x_self.shape
        H, d_h = self.num_heads, self.head_dim

        # ---- Per-head projections ----
        # s0 = self.proj_self(x_self)
        i0 = self.proj_img(x_img)
        c0 = self.proj_con(x_con)

        # ---- Contextual gating ----
        ctx = torch.cat([x_img, x_con], dim=-1)
        gates = self.gate_net(ctx)                         # [B,L,3D]
        gates = gates.view(B, L, H, 2, d_h)

        # Temperature-scaled softmax
        weights = F.softmax(gates / self.temperature, dim=3)

        # w_self = weights[:, :, :, 0]
        w_img  = weights[:, :, :, 0]
        w_con  = weights[:, :, :, 1]

        # ---- Reshape modalities ----
        # s0 = s0.view(B, L, H, d_h)
        i0 = i0.view(B, L, H, d_h)
        c0 = c0.view(B, L, H, d_h)

        # ---- Weighted fusion ----
        fused = w_img * i0 + w_con * c0
        fused = fused.view(B, L, D)

        # ---- Projection + residual + normalization ----
        out = x_self + self.out_proj(fused)
        out = self.norm(out)

        # ---- Extra FFN ----
        out = out + self.ffn(out)

        return out, weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PAM(nn.Module):
    def __init__(self, dim=512):
        super(PAM, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 13, 1, 13//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        B, H, C = x.shape
        # print(f'B, H, C : {(B, H, C)}')
        assert int(math.sqrt(H))**2==H, f'{x.shape}'
        cnn_feat = x.transpose(1, 2).reshape(B, C, int(math.sqrt(H)), int(math.sqrt(H)))
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)

        return x

class PAM_(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),)

    def forward(self, x):
        # B, H, C = x.shape
        x = self.proj(x)

        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, n_heads,d_model, dropout):
        super().__init__()
        self.cross_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.norm = LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, 4 * d_model, dropout)

    def forward(self, x, concepts):
        x2, attn_x2c = self.cross_attn(x, concepts, concepts)
        c2, _ = self.cross_attn(concepts, x, x)
        c2_to_x = torch.matmul(attn_x2c.mean(1), c2)
        x_fused = self.norm(x + self.ff(x2 + c2_to_x))
        return x_fused

class ConceptInfusionBlockV2(nn.Module):
    """
    Stable bidirectional concept infusion with:
      - shared Q/K/V projections
      - lightweight concept update
      - safe pooled concept context injection
      - cooperative gating (not competitive)
    """

    def __init__(self, n_heads, d_model, dropout=0.1, concept_update=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.concept_update = concept_update

        # ---- 1) Shared cross-attention projections -------------------------
        self.x_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.c_attn = MultiHeadedAttention(n_heads, d_model, dropout)

        # ---- 2) Feed-forwards ----------------------------------------------
        self.ff_x = PositionwiseFeedForward(d_model, 4 * d_model, dropout)
        self.ff_c = PositionwiseFeedForward(d_model, 4 * d_model, dropout)

        # ---- 3) Norms ------------------------------------------------------
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)

        # ---- 4) Concept pooling (learned) ---------------------------------
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

        # ---- 5) Cooperative gate (only scales concept signal) ------------
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # ---- 6) Output projection -----------------------------------------
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, concepts):
        """
        x:        [B, L, D]
        concepts: [B, M, D]
        """

        # ----------- A) Main: X attends to Concept Tokens -------------------
        x2c, att_x2c = self.x_attn(x, concepts, concepts)   # X <- Concepts
        x = self.norm_x(x + x2c)
        x = self.ff_x(x)

        # ----------- B) Optional Concept Update -----------------------------
        if self.concept_update:
            c2x, att_c2x = self.c_attn(concepts, x, x)  # C <- X
            concepts = self.norm_c(concepts + c2x)
            concepts = self.ff_c(concepts)
        else:
            att_c2x = None

        # ----------- C) Pool Concepts to a Small Context Vector ------------
        # [B, M, D] → [B, 1, D]
        c_pooled = self.pool(concepts.mean(dim=1, keepdim=True))

        # ----------- D) Cooperative Fusion ---------------------------------
        # gate ∈ [0,1] expands concept info but never suppresses x
        g = self.gate(x)                       # [B, L, D]
        fused = x + g * c_pooled               # additive, cooperative
        fused = self.out_proj(fused)

        return fused, att_x2c, att_c2x



class EncoderDecoder(AttModel):

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        tgt_vocab = self.vocab_size + 1

        # self.embeded = Embeddings(args.d_vf, tgt_vocab)
        self.model = self.__build_model(tgt_vocab)
        self.__init_model()

        self.logit = nn.Linear(args.d_model, tgt_vocab)
        # self.logit_mesh = nn.Linear(args.d_model, args.d_model)

    def __build_model(self, tgt_vocab):
        attn = MultiHeadedAttention(self.num_heads, self.d_model, dropout=self.dropout)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        pp = PAM(self.d_model)
        mgf = MultiHeadGatedFusionV3(self.d_model, self.num_heads, dropout=self.dropout)
        concept_fusion = ConceptInfusionBlockV2(self.num_heads, self.d_model, dropout=self.dropout)

        model = Transformer(
            Encoder(
                EncoderLayer(
                    self.d_model,
                    deepcopy(attn),
                    deepcopy(ff),
                    self.dropout
                ),
                self.num_layers, pp, concept_fusion
            ),
            Decoder(
                DecoderLayer(
                    self.d_model,
                    deepcopy(attn),  # self-attn
                    clones(attn, 3),     # src-attns [patch, slide, concept]
                    deepcopy(ff),  # feed-forward
                    deepcopy(mgf), # multi head gate fusion
                    self.dropout
                ),
                self.num_layers
            ),
            LayerNorm(self.d_model),
            # Target token embedding + position
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), deepcopy(position)),
            # Concept embedding module
            LayerNorm(self.d_model),
            LayerNorm(self.d_model)
        )
        return model


    def init_hidden(self, bsz):
        return []

    def __init_model(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _prepare_feature(self, fc_feats, att_feats, att_masks, slide_embeddings,concept_embeddings, meshes=None):
        att_feats = pad_tokens(att_feats)
        att_feats, seq, _, att_masks, seq_mask, _ = self._prepare_feature_forward(
            att_feats, att_masks, meshes
        )

        memory = self.model.encode(att_feats, slide_embeddings,concept_embeddings, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_mesh(self, att_feats, att_masks=None, meshes=None):
        att_feats = pad_tokens(att_feats)
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # gc_feats = pack_wrapper(self.gc_embed, gc_feats)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if meshes is not None:
            # crop the last one
            meshes = meshes[:, :-1]
            meshes_mask = (meshes.data > 0)
            meshes_mask[:, 0] += True

            meshes_mask = meshes_mask.unsqueeze(-2)
            meshes_mask = meshes_mask & subsequent_mask(meshes.size(-1)).to(meshes_mask)
        else:
            meshes_mask = None

        return att_feats, meshes, att_masks, meshes_mask

    def _prepare_feature_forward(self, att_feats, att_masks=None, meshes=None, seq=None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        else:
            print(f'att_masks: {att_masks.shape}')
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        if meshes is not None:
            # crop the last one
            meshes = meshes[:, :-1]
            meshes_mask = (meshes.data > 0)
            meshes_mask[:, 0] += True

            meshes_mask = meshes_mask.unsqueeze(-2)
            meshes_mask = meshes_mask & subsequent_mask(meshes.size(-1)).to(meshes_mask)
        else:
            meshes_mask = None

        return att_feats, seq, meshes, att_masks, seq_mask, meshes_mask

    def _forward(self, fc_feats, att_feats, slide_embeddings,concept_embeddings, report_ids, att_masks=None):
        # log_message(fc_feats, att_feats, report_ids, att_masks)
        att_feats, report_ids, att_masks, report_mask = self._prepare_feature_mesh(
            att_feats, att_masks, report_ids
        )
        # print(f'att_masks: {att_masks}')
        out, attn_maps = self.model(att_feats, slide_embeddings,concept_embeddings, report_ids, att_masks, report_mask)

        # print(f'out: {out}')
        outputs = F.log_softmax(self.logit(out), dim=-1)
        # print(f'outputs: {outputs}')

        return outputs, attn_maps

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.long().unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out, attn_maps = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)], attn_maps

    def _encode(self, fc_feats, gc_feats, att_feats, att_masks=None):

        att_feats, gc_feats, _, att_masks, _ = self._prepare_feature_mesh(att_feats, gc_feats, att_masks)
        out = self.model.encode(att_feats,gc_feats, att_masks)
        return out