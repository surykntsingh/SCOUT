import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer import EncoderDecoder

class ConceptEmbeddingSupervisionHead(nn.Module):
    """
    Supervise decoder concept-stream (x_con) to match Gecko concept activations.
    x_con: [B, L, D]
    concept_scores: [B, C]
    """
    def __init__(self, d_model, concept_dim, hidden=256, mode="mse", temp=0.07):
        super().__init__()
        self.mode = mode
        self.temp = temp
        self.concept_dim = concept_dim

        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, concept_dim)
        )

        self.cos = nn.CosineSimilarity(dim=-1)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x_con, concept_scores):
        """
        x_con: [B, L, D]
        concept_scores: [B, C]
        """
        # B, L, D = x_con.shape
        assert concept_scores.shape[-1] == self.concept_dim

        # ---- Pool over tokens ----
        pooled = x_con.mean(dim=1)     # [B, D]

        # ---- Predict concept scores ----
        pred = self.proj(pooled)       # [B, C]

        # ---- Choose supervision type ----
        if self.mode == "mse":
            loss = self.mse(pred, concept_scores)

        elif self.mode == "bce":
            loss = self.bce(pred, concept_scores)

        elif self.mode == "cosine":
            pred_n = F.normalize(pred, dim=-1)
            tgt_n  = F.normalize(concept_scores, dim=-1)
            loss = 1 - self.cos(pred_n, tgt_n).mean()

        elif self.mode == "kl":
            log_p = F.log_softmax(pred / self.temp, dim=-1)
            q = F.softmax(concept_scores / self.temp, dim=-1)
            loss = self.kl(log_p, q)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return loss, pred



# class ConceptSupervisionHead(nn.Module):
#     def __init__(self, d_model, concept_dim,dropout):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(d_model, concept_dim),
#             # nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(concept_dim, concept_dim)
#         )
#
#         self.cosine = nn.CosineSimilarity(dim=-1)
#
#     def forward(self, decoder_out, gecko_concepts):
#         # decoder_out: (batch, seq_len, d_model)
#         # gecko_concepts: (batch, num_concepts, concept_dim)
#
#         pred = self.proj(decoder_out)  # mean over tokens
#         # gecko_mean = gecko_concepts.mean(dim=1)
#         pred_norm = F.normalize(pred, dim=-1)
#         gecko_norm = F.normalize(gecko_concepts, dim=-1)
#         # print(f'decoder_out: {pred_norm.shape}, gecko_concepts: {gecko_norm.shape}')
#         return 1 - self.cosine(pred_norm, gecko_norm).mean()
#
# class ConceptHead(nn.Module):
#     def __init__(self, d_model, concept_dim,dropout):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model//2, 1)
#         )
#         self.adapter = nn.Sequential(
#             nn.Linear(d_model, concept_dim),
#         )
#
#     def forward(self, fused_concepts, target_concepts):
#         # fused_concepts:
#         # print(f'fused_concepts: {fused_concepts.shape} target_concepts: {target_concepts.shape}')
#         preds = self.proj(fused_concepts).squeeze(-1) # [B, seq, 1]
#
#         # print(f'preds: {preds.shape}, target_concepts: {target_concepts.shape}')
#         loss = F.mse_loss(preds, target_concepts)
#         # print(f'concept_loss {loss}')
#         return loss

class ConceptEncoder(nn.Module):
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

class ReportGenModel(nn.Module):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.__tokenizer = tokenizer

        self.prompt = nn.Parameter(torch.randn(1, 1, args.d_vf))

        d = args.d_vf
        self.encoder = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.LayerNorm(2*d),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(d, d)
        )
        d1 = args.d1
        d2 = args.d2
        self.adapter_mlp_1 = nn.Sequential(
            nn.Linear(d1, 2 * d1),
            nn.ReLU(),
            nn.Linear(2 * d1, 2*d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(2*d, d)
        )

        self.adapter_mlp_2 = nn.Sequential(
            nn.Linear(d2, 2 * d2),
            nn.ReLU(),
            nn.Linear(2 * d2, 2 * d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(2 * d, d)
        )

        self.concept_encoder = ConceptEncoder(args.gcd, args.d_model, args.dropout_mlp)
        self.slide_encoder = ChannelProjector(args.d1, args.d_model, args.dropout_mlp)
        self.gecko_projector = ChannelProjector(args.gcd, args.d_model, args.dropout_mlp)
        self.gecko_deep_projector = ChannelProjector(args.gd, args.d_model, args.dropout_mlp)
        self.concept_supervision_head = ConceptEmbeddingSupervisionHead(args.d_model, args.gcd, args.dropout_mlp)

        # gd = args.gd
        dm =args.d_model
        self.gecko_mlp = nn.Sequential(
            nn.Linear(dm, 2 * dm),
            nn.ReLU(),
            nn.Linear(2 * dm, 4 * dm),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(4 * dm, 2*dm),
            nn.ReLU(),
            nn.Linear(2 * dm, dm),
        )

        self.gecko_encoder = nn.Sequential(
            nn.Linear(dm, 2 * dm),
            nn.ReLU(),
            nn.Linear(2 * dm, 4 * dm),
            nn.ReLU(),
            nn.Linear(4 * dm, 2*dm),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(2*dm, dm)
        )

        self.encoder_decoder = EncoderDecoder(args, tokenizer)


    def freeze_deep_features(self):
        print('Freezing concept parameters')
        for param in self.concept_encoder.parameters():
            param.requires_grad = False

        for param in self.concept_supervision_head.parameters():
            param.requires_grad = False

        for param in self.encoder_decoder.model.concept_embed.parameters():
            param.requires_grad = False

        for param in self.encoder_decoder.gc_embed.parameters():
            param.requires_grad = False


    def forward(self, image_embeddings1, image_embeddings2, emb_g, emb_gc, attn_gc, report_ids=None, patch_masks=None, mode='train'):
        # coords_encoded = self.positional_encoder(pos_embeddings)
        # patch_feats = image_embeddings # + coords_encoded
        # print(f'image_embeddings1: {image_embeddings1}')
        patch_masks=None
        image_embeddings1 = self.adapter_mlp_1(self.slide_encoder(image_embeddings1))
        image_embeddings2 = self.adapter_mlp_2(image_embeddings2)

        emb_gc_proj = self.gecko_mlp(self.gecko_projector(emb_gc))

        gecko_embeddings = self.gecko_encoder(torch.cat([self.gecko_deep_projector(emb_g),emb_gc_proj], dim=1))

        # gecko_embeddings = self.gecko_encoder(emb_g)

        patch_feats = torch.cat([image_embeddings1, image_embeddings2, gecko_embeddings], dim=1)
        # patch_feats = self.encoder(gecko_embeddings)
        att_feats = torch.cat([self.prompt, patch_feats], dim=1)
        # att_feats = self.prompt
        fc_feats = torch.sum(att_feats, dim=1)
        attn_gc = self.concept_encoder(attn_gc)

        if mode == 'train':
            output, concept_attn_maps, concept_tokens = self.encoder_decoder(fc_feats, att_feats, attn_gc, report_ids, mode='forward')
        elif mode == 'sample':
            output, _, concept_attn_maps, concept_tokens = self.encoder_decoder(fc_feats, att_feats, attn_gc, mode='sample')
        elif mode == 'encode':
            output = self.encoder_decoder(fc_feats, att_feats, attn_gc, mode='encode')

            logits = self.fc(output[0, 0, :]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
        else:
            raise ValueError

        return output, concept_attn_maps, concept_tokens