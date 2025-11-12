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



class ConceptHead(nn.Module):
    def __init__(self, d_model, concept_dim,dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )
        self.adapter = nn.Sequential(
            nn.Linear(d_model, concept_dim),
        )

    def forward(self, fused_concepts, target_concepts):
        # fused_concepts:
        # print(f'fused_concepts: {fused_concepts.shape} target_concepts: {target_concepts.shape}')
        preds = self.proj(fused_concepts).squeeze(-1) # [B, seq, 1]

        # print(f'preds: {preds.shape}, target_concepts: {target_concepts.shape}')
        loss = F.mse_loss(preds, target_concepts)
        # print(f'concept_loss {loss}')
        return loss