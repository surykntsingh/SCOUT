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
        self.encoder = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, 4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, 2 * d),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(d, d)
        )
        d1 = args.d1
        d2 = args.d2
        self.adapter_mlp = nn.Sequential(
            nn.Linear(d1, 2 * d1),
            nn.ReLU(),
            nn.Linear(2 * d1, 2 * d2),
            nn.ReLU(),
            nn.Linear(2 * d2, d2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(d2, d)
        )

        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    def forward(self, image_embeddings1, image_embeddings2, report_ids, patch_masks, mode='train'):
        # coords_encoded = self.positional_encoder(pos_embeddings)
        # patch_feats = image_embeddings # + coords_encoded
        # print(f'image_embeddings1: {image_embeddings1}')
        image_embeddings1 = self.adapter_mlp(image_embeddings1)
        print(f'image_embeddings1: {image_embeddings1.shape}, image_embeddings2: {image_embeddings2.shape}')
        patch_feats = torch.cat([image_embeddings1, image_embeddings2], dim=1)
        patch_feats = self.encoder(patch_feats)
        att_feats = torch.cat([self.prompt, patch_feats], dim=1)
        fc_feats = torch.sum(att_feats, dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, report_ids, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        elif mode == 'encode':
            output = self.encoder_decoder(fc_feats, att_feats, mode='encode')

            logits = self.fc(output[0, 0, :]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
        else:
            raise ValueError

        return output