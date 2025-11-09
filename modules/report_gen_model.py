import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer import EncoderDecoder

class ConceptEncoder(nn.Module):
    def __init__(self, n_concepts, d_model, dropout, hidden=256):
        super().__init__()
        # Option A: per-concept learned embedding table (concept id -> vector)
        self.id_embed = nn.Embedding(n_concepts, d_model)
        # Projection from scalar activation (score) to scale per concept
        self.score_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
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

        gd = args.gd
        gcd =args.gcd
        self.gecko_mlp = nn.Sequential(
            nn.Linear(gcd, 2 * gcd),
            nn.ReLU(),
            nn.Linear(2 * gcd, 4 * gcd),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(4 * gcd, 2*gd),
            nn.ReLU(),
            nn.Linear(2 * gd, gd),
        )

        self.gecko_encoder = nn.Sequential(
            nn.Linear(gd, 2 * gd),
            nn.ReLU(),
            nn.Linear(2 * gd, 4 * gd),
            nn.ReLU(),
            nn.Linear(4 * gd, 2*d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(2*d, d)
        )

        self.encoder_decoder = EncoderDecoder(args, tokenizer)


    def freeze_deep_features(self):
        for param in self.encoder_decoder.parameters():
            param.requires_grad = False




    def forward(self, image_embeddings1, image_embeddings2, emb_g, emb_gc, attn_gc, report_ids=None, patch_masks=None, mode='train'):
        # coords_encoded = self.positional_encoder(pos_embeddings)
        # patch_feats = image_embeddings # + coords_encoded
        # print(f'image_embeddings1: {image_embeddings1}')
        patch_masks=None
        image_embeddings1 = self.adapter_mlp_1(image_embeddings1)
        image_embeddings2 = self.adapter_mlp_2(image_embeddings2)

        emb_gc_proj = self.gecko_mlp(emb_gc.unsqueeze(1))

        gecko_embeddings = self.gecko_encoder(torch.cat([emb_g,emb_gc_proj], dim=1))

        # gecko_embeddings = self.gecko_encoder(emb_g)

        patch_feats = torch.cat([image_embeddings1, image_embeddings2, gecko_embeddings], dim=1)
        patch_feats = self.encoder(patch_feats)
        att_feats = torch.cat([self.prompt, patch_feats], dim=1)
        fc_feats = torch.sum(att_feats, dim=1)
        attn_gc = self.concept_encoder(emb_gc)
        encoded_mm_tokens =None

        if mode == 'train':
            output, concept_attn_maps, encoded_mm_tokens = self.encoder_decoder(fc_feats, att_feats, attn_gc, report_ids, mode='forward')
        elif mode == 'sample':
            output, _, concept_attn_maps = self.encoder_decoder(fc_feats, att_feats, attn_gc, mode='sample')
        elif mode == 'encode':
            output = self.encoder_decoder(fc_feats, att_feats, attn_gc, mode='encode')

            logits = self.fc(output[0, 0, :]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
        else:
            raise ValueError

        return output, concept_attn_maps, attn_gc, encoded_mm_tokens