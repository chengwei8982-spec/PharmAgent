import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.ban import MLP

class PromptAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        """
        Attention-based prompt fusion module
        Args:
            dim: The dimension of prompt features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Multi-head attention layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim) 
        self.v_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, num_prompts, dim)
        Returns:
            Fused prompt representation of shape (batch_size, dim)
        """
        # Save original input for residual
        identity = x
        
        # Project to Q, K, V
        B, N, D = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D/H)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, N, D/H)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.out_proj(out)
        
        # First residual connection and layer norm
        out = self.norm1(identity + out)
        
        # Feed-forward network
        ff_out = self.ffn(out)
        
        # Second residual connection and layer norm
        out = self.norm2(out + ff_out)
        
        # Average pooling across prompts
        out = out.mean(dim=1)  # (B, D)
        
        return out

    def get_attention_weights(self, x):
        """
        Get attention weights for visualization
        Args:
            x: Input tensor of shape (batch_size, num_prompts, dim)
        Returns:
            Attention weights of shape (batch_size, num_heads, num_prompts, num_prompts)
        """
        B, N, D = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        return attn  # (B, H, N, N)


class TwoStagePromptFusion(nn.Module):
    def __init__(self, dim,dim_out, phar_num_list, dropout=0.1,projection_layers=2,d_hidden_feats=256):
        super().__init__()
        self.phar_num_list = phar_num_list
        self.projection_layers = projection_layers
        self.d_hidden_feats = d_hidden_feats

        # inner compress
        self.pharma_projection_model_list = nn.ModuleList()
        for pharma_index in range(len(phar_num_list)):
            self.pharma_projection_model_list.append(
                MLP(phar_num_list[pharma_index] * dim,
                                dim_out, self.projection_layers,
                                nn.GELU(), d_hidden_feats=self.d_hidden_feats, dropout=dropout))
        # inter compress
        self.prompt_projection_model = MLP(len(phar_num_list) * dim_out,
                                            dim_out, self.projection_layers,
                                            nn.GELU(), d_hidden_feats=self.d_hidden_feats, dropout=dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]
        start = 0
        molecules_phar_prompt = []
        for pharma_index in range(len(self.phar_num_list)):
            molecules_phar_prompt.append(self.pharma_projection_model_list[pharma_index](
                x[:, start:start + self.phar_num_list[pharma_index], :].reshape(batch_size, -1)))
            start += self.phar_num_list[pharma_index]

        molecules_prompt = torch.stack(molecules_phar_prompt, dim=1)
        molecules_prompt = self.prompt_projection_model(molecules_prompt.reshape(batch_size, -1))  # [B, D]

            
        # directly return the stack feature
        return molecules_prompt  # [B,D] 
    
class OneStagePromptFusion(nn.Module):
    def __init__(self):
        super().__init__()
                
    def forward(self, x):
        molecules_prompt = x.mean(dim=1)  # [B,N, D] -> [B,D]
        return molecules_prompt  # [B,D] 
    
class StaticPharmAgentFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x  # [B,D] 