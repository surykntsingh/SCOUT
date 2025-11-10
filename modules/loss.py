import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class ConceptSupervisionHead(nn.Module):
    def __init__(self, d_model, concept_dim,dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, concept_dim),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(concept_dim, concept_dim)
        )

        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, decoder_out, gecko_concepts):
        # decoder_out: (batch, seq_len, d_model)
        # gecko_concepts: (batch, num_concepts, concept_dim)

        pred = self.proj(decoder_out)  # mean over tokens
        # gecko_mean = gecko_concepts.mean(dim=1)
        pred_norm = F.normalize(pred, dim=-1)
        gecko_norm = F.normalize(gecko_concepts, dim=-1)
        # print(f'decoder_out: {pred_norm.shape}, gecko_concepts: {gecko_norm.shape}')
        return 1 - self.cosine(pred_norm, gecko_norm).mean()

class ConceptHead(nn.Module):
    def __init__(self, d_model, concept_dim,dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )
        # self.adapter = nn.Sequential(
        #     nn.Linear(d_model, concept_dim),
        # )

    def forward(self, fused_concepts, target_concepts):
        # fused_concepts:
        # print(f'fused_concepts: {fused_concepts.shape} target_concepts: {target_concepts.shape}')
        preds = self.proj(fused_concepts).squeeze(-1) # [B, seq, 1]

        # print(f'preds: {preds.shape}, target_concepts: {target_concepts.shape}')
        loss = F.mse_loss(preds, target_concepts)
        # print(f'concept_loss {loss}')
        return loss