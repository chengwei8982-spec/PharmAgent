import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None, dropout=0.1):
        super().__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dropout = nn.Dropout(p=dropout)
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(self.d_hidden_feats, self.d_hidden_feats) for _ in range(max(self.n_dense_layers - 2, 0))
        )
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats):
        feats = self.dropout(self.act(self.in_proj(feats)))
        for layer in self.hidden_layers:
            feats = self.dropout(self.act(layer(feats)))
        return self.out_proj(feats)


class FCNet_ca(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0.0, norm=None, residual=False, init='kaiming', final_act=False, dropout_after_act=True):
        super().__init__()
        self.residual = residual
        act_dict = {'ReLU': nn.ReLU(), 'GELU': nn.GELU(), 'LeakyReLU': nn.LeakyReLU(0.2), 'Swish': nn.SiLU()}
        activation = act_dict.get(act, nn.ReLU()) if isinstance(act, str) else act
        norm_layers = {'BatchNorm': nn.BatchNorm1d, 'LayerNorm': nn.LayerNorm, None: None}
        norm_layer = norm_layers.get(norm, None)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if norm_layer is not None:
                layers.append(norm_layer(out_dim))
            if (i != len(dims) - 2) or final_act:
                if activation is not None:
                    layers.append(activation)
                if dropout > 0 and dropout_after_act:
                    layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        if self.residual and out.shape == x.shape:
            return out + x
        return out


class KBANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, k_dim, h_out, act=nn.ReLU(), dropout=0.2, k=3, c=32, fusion_method='vk'):
        super().__init__()
        self.h_dim = h_dim
        self.h_out = h_out
        self.v_proj = nn.Linear(v_dim, h_dim)
        self.q_proj = nn.Linear(q_dim, h_dim)
        self.k_proj = nn.Linear(k_dim, h_dim)
        self.fusion = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
        )

    @staticmethod
    def _masked_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        weight = mask.float().unsqueeze(-1)
        denom = weight.sum(dim=1).clamp(min=1.0)
        return (x * weight).sum(dim=1) / denom

    def forward(self, v, q, k, v_mask=None, q_mask=None, k_mask=None, softmax=False):
        v_proj = self.v_proj(v)
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)

        v_summary = self._masked_mean(v_proj, v_mask)
        q_summary = self._masked_mean(q_proj, q_mask)
        k_summary = self._masked_mean(k_proj, k_mask)

        logits = self.fusion(torch.cat([v_summary, q_summary, k_summary], dim=-1))

        att_scores = torch.matmul(v_proj, q_proj.transpose(1, 2)) / (self.h_dim ** 0.5)
        if v_mask is not None:
            att_scores = att_scores.masked_fill(v_mask.unsqueeze(-1) == 0, -1e9)
        if q_mask is not None:
            att_scores = att_scores.masked_fill(q_mask.unsqueeze(1) == 0, -1e9)
        att_map = F.softmax(att_scores, dim=-1) if softmax else att_scores

        return logits, att_map, None, None
