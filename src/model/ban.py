import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None, dropout=0.1):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats):
        feats = self.dropout(self.act(self.in_proj(feats)))
        for i in range(self.n_dense_layers - 2):
            feats = self.dropout(self.act(self.dense_layer_list[i](feats)))
        feats = self.out_proj(feats)
        return feats
    

def compute_attention_mask(v_mask, q_mask, att_maps, h_out, num_v=None, num_q=None):
    atten_mask = []
    num_samples = v_mask.size()[0]

    for i in range(num_samples):
        v_mask_float = v_mask[i].float().reshape(-1, 1)
        q_mask_float = q_mask[i].float().reshape(1, -1)
        atten_mask.append(torch.matmul(v_mask_float, q_mask_float))

    atten_mask = torch.stack(atten_mask)
    atten_mask = atten_mask.unsqueeze(1).repeat(1, h_out, 1, 1)

    att_maps = att_maps - (1 - atten_mask) * 1e9

    p = F.softmax(att_maps.view(-1, h_out, num_v * num_q), 2)

    return p.view(-1, h_out, num_v, num_q)


class KBANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, k_dim, h_out, act=nn.ReLU(), dropout=0.2, k=3, c=32, fusion_method='vk'):
        super(KBANLayer, self).__init__()

        self.c = c
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.h_dim = h_dim
        self.h_out = h_out
        self.fusion_method = fusion_method

        # 改进后的特征提取网络
        self.v_net = FCNet_ca([v_dim, h_dim * self.k], act='GELU', norm='LayerNorm', residual=True)
        self.q_net = FCNet_ca([q_dim, h_dim * self.k], act='Swish', dropout=dropout)
        self.k_net = nn.Sequential(
            FCNet_ca([k_dim, h_dim * 2], act='GELU', norm='LayerNorm', residual=True),
            FCNet_ca([h_dim * 2, h_dim * self.k], act='GELU', dropout=dropout)
        )

        if self.fusion_method == 'qk' or self.fusion_method == 'vk':
            self.k_cross_attn = nn.MultiheadAttention(embed_dim=h_dim * self.k, num_heads=4)
        elif self.fusion_method == 'bi_qk' or self.fusion_method == 'bi_vk':
            self.h_mat_qk = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias_qk = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())

            self.k_cross_attn_f = nn.MultiheadAttention(embed_dim=h_dim * self.k, num_heads=4)
            self.k_cross_attn_b = nn.MultiheadAttention(embed_dim=h_dim * self.k, num_heads=4)

        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat_vq = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias_vq = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())

            nn.init.xavier_normal_(self.h_mat_vq)
            nn.init.kaiming_uniform_(self.h_bias_vq, nonlinearity='linear')

        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, k, v_mask=None, q_mask=None, k_mask=None, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        k_num = k.size(1)

        v_ = self.v_net(v)
        q_ = self.q_net(q)
        k_ = self.k_net(k)

        # fusion method
        if self.fusion_method == 'vk':
            k_cross, _ = self.k_cross_attn(
                query=v_.transpose(0, 1),
                key=k_.transpose(0, 1),
                value=k_.transpose(0, 1)
            )
            v_ = v_ + k_cross.transpose(0, 1)  # 残差连接
        elif self.fusion_method == 'qk':
            k_cross, _ = self.k_cross_attn(
                query=q_.transpose(0, 1),
                key=k_.transpose(0, 1),
                value=k_.transpose(0, 1)
            )
            q_ = q_ + k_cross.transpose(0, 1)  # 残差连接
        elif self.fusion_method == 'bi_qk':
            q_fused = self.k_cross_attn_f(query=q_.transpose(0, 1), key=k_.transpose(0, 1), value=k_.transpose(0, 1))[
                0].transpose(0, 1)
            k_fused = self.k_cross_attn_b(query=k_.transpose(0, 1), key=q_.transpose(0, 1), value=q_.transpose(0, 1))[
                0].transpose(0, 1)
            # Key/Value由融合后的q和k拼接组成
            q_ = torch.cat([q_fused, k_fused], dim=1)  # [B, q_len+k_len, h]
            q_mask = torch.cat([q_mask, k_mask], dim=-1)

        att_maps_vq = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat_vq, v_, q_)) + self.h_bias_vq

        if softmax:
            att_maps_vq = compute_attention_mask(v_mask, q_mask, att_maps_vq, self.h_out, v_num, q_num)

        logits_vq = self.attention_pooling(v_, q_, att_maps_vq[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_vq = logits_vq + self.attention_pooling(v_, q_, att_maps_vq[:, i, :, :])

        logits_vq = self.bn(logits_vq)
        logits = logits_vq
        return logits, att_maps_vq[:, -1, :, :], None, None,  # att_maps_vk[:, -1, :, :], att_maps_qk[:, -1, :, :]


class FCNet_ca(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0.0,
                 norm=None, residual=False, init='kaiming',
                 final_act=False, dropout_after_act=True):
        super().__init__()
        self.residual = residual
        layers = []

        act_dict = {'ReLU': nn.ReLU(), 'GELU': nn.GELU(),
                    'LeakyReLU': nn.LeakyReLU(0.2), 'Swish': nn.SiLU()}
        if isinstance(act, str):
            act = act_dict.get(act, nn.ReLU())

        # 归一化层
        norm_layers = {
            'BatchNorm': nn.BatchNorm1d,
            'LayerNorm': nn.LayerNorm,
            None: None
        }
        norm_layer = norm_layers.get(norm, None)

        # 构建网络
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))

            # 初始化
            if init == 'kaiming':
                kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='relu')
            elif init == 'xavier':
                nn.init.xavier_normal_(layers[-1].weight)

            # 归一化
            if norm_layer is not None:
                layers.append(norm_layer(out_dim))

            # 激活函数和 Dropout
            if (i != len(dims) - 2) or final_act:
                if act is not None:
                    layers.append(act)
                if dropout > 0 and dropout_after_act:
                    layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        if self.residual and out.shape == x.shape:
            return out + x
        return out


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(act)
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(act)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits
