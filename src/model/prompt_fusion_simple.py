import torch
import torch.nn as nn

from src.model.simple_fusion import MLP


class PromptAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj = MLP(dim, dim, 2, nn.GELU(), d_hidden_feats=dim, dropout=dropout)

    def forward(self, x):
        return self.proj(x.mean(dim=1))

    def get_attention_weights(self, x):
        batch_size, num_prompts, _ = x.shape
        return torch.full((batch_size, 1, num_prompts, num_prompts), 1.0 / max(num_prompts, 1), device=x.device)


class TwoStagePromptFusion(nn.Module):
    def __init__(self, dim, dim_out, phar_num_list, dropout=0.1, projection_layers=2, d_hidden_feats=256):
        super().__init__()
        self.output_projection = MLP(dim, dim_out, projection_layers, nn.GELU(), d_hidden_feats=d_hidden_feats, dropout=dropout)

    def forward(self, x):
        return self.output_projection(x.mean(dim=1))


class OneStagePromptFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)


class StaticPharmAgentFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
