import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformer import EncoderDecoder


class ReportGenModel(nn.Module):

    def __init__(self, args, tokenizer):
        super().__init__()
        self.__tokenizer = tokenizer

        self.prompt = nn.Parameter(torch.randn(1, 1, args.d_vf))

        d = args.d_vf
        self.slide_encoder = nn.Sequential(
            nn.Linear(2 * d,2*d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        d1 = args.d1
        d2 = args.d2
        gd = args.gd
        gcd = args.gcd
        self.mlp_slide_adapter = nn.Sequential(
            nn.Linear(d1, 2 * d1),
            nn.ReLU(),
            nn.Linear(2 * d1, 2*d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(2*d, d)
        )

        self.mlp_patch_adapter = nn.Sequential(
            nn.Linear(d2, 2 * d2),
            nn.ReLU(),
            nn.Linear(2 * d2, 2 * d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )

        self.mlp_gecko_deep_adapter = nn.Sequential(
            nn.Linear(gd, 2 * gd),
            nn.ReLU(),
            nn.Linear(2 * gd, 2 * d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(2 * d, d)
        )

        self.mlp_gecko_concept_adapter = nn.Sequential(
            nn.Linear(gcd, 2 * gcd),
            nn.ReLU(),
            nn.Linear(2 * gcd, 2 * d),
            nn.ReLU(),
            nn.Dropout(args.dropout_mlp),
            nn.Linear(2 * d, d)
        )

        # self.concept_encoder = ConceptEncoder(args.gcd, args.d_model, args.dropout_mlp)
        # self.slide_encoder = ChannelProjector(args.d1, args.d_model, args.dropout_mlp)
        # self.gecko_projector = ChannelProjector(args.gcd, args.d_model, args.dropout_mlp)
        # self.gecko_deep_projector = ChannelProjector(args.gd, args.d_model, args.dropout_mlp)
        # self.concept_supervision_head = ConceptSupervisionHead(args.d_model, args.gcd, args.dropout_mlp)

        # gd = args.gd
        # dm =args.d_model
        # self.gecko_mlp = nn.Sequential(
        #     nn.Linear(dm, 2 * dm),
        #     nn.ReLU(),
        #     nn.Linear(2 * dm, 4 * dm),
        #     nn.ReLU(),
        #     nn.Dropout(args.dropout_mlp),
        #     nn.Linear(4 * dm, 2*dm),
        #     nn.ReLU(),
        #     nn.Linear(2 * dm, dm),
        # )
        #
        # self.gecko_encoder = nn.Sequential(
        #     nn.Linear(dm, 2 * dm),
        #     nn.ReLU(),
        #     nn.Linear(2 * dm, 4 * dm),
        #     nn.ReLU(),
        #     nn.Linear(4 * dm, 2*dm),
        #     nn.ReLU(),
        #     nn.Dropout(args.dropout_mlp),
        #     nn.Linear(2*dm, dm)
        # )

        self.encoder_decoder = EncoderDecoder(args, tokenizer)


    def forward(self, features, report_ids=None, mode='train'):

        patch_embeddings = self.mlp_patch_adapter(features['patch'])
        slide_embeddings = self.mlp_slide_adapter(features['slide'])
        gecko_deep_embeddings = self.mlp_gecko_deep_adapter(features['gecko']['deep'])
        slide_embeddings = self.slide_encoder(torch.cat([slide_embeddings,gecko_deep_embeddings], dim=-1))
        concept_embeddings = self.mlp_gecko_concept_adapter(features['gecko']['concept'])

        att_feats = torch.cat([self.prompt, patch_embeddings], dim=1)
        # att_feats = self.prompt
        fc_feats = torch.sum(att_feats, dim=1)

        if mode == 'train':
            output, attn_maps = self.encoder_decoder(fc_feats, att_feats, slide_embeddings,concept_embeddings , report_ids, mode='forward')
        elif mode == 'sample':
            output, _, attn_maps = self.encoder_decoder(fc_feats, att_feats, slide_embeddings,concept_embeddings, mode='sample')
        elif mode == 'encode':
            output = self.encoder_decoder(fc_feats, att_feats, slide_embeddings,concept_embeddings, mode='encode')

            logits = self.fc(output[0, 0, :]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
        else:
            raise ValueError

        return output, attn_maps