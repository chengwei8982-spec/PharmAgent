import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dgl.data.utils import load_graphs

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from src.data.collator import Collator_pharmaPrompt
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import PharmaQADataset
from src.model.ban import FCNet_ca, KBANLayer, MLP, compute_attention_mask
from src.model.light import LiGhTPredictor as LiGhT, TextEncoder
from src.model.prompt_fusion import TwoStagePromptFusion
from src.model_config import config_dict

phar_num_list = [1, 1, 1, 4, 6, 5, 2, 7]


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda m: init_params(m))
    return predictor.to(device)


class LegacyAceKBANLayer(nn.Module):
    """Compatibility layer for older ACE checkpoints with explicit vk/qk attention params."""

    def __init__(self, v_dim, q_dim, h_dim, k_dim, h_out, act=nn.ReLU(), dropout=0.2, k=3, c=32, fusion_method="vk"):
        super().__init__()
        self.c = c
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.h_dim = h_dim
        self.h_out = h_out
        self.fusion_method = fusion_method

        self.v_net = FCNet_ca([v_dim, h_dim * self.k], act="GELU", norm="LayerNorm", residual=True)
        self.q_net = FCNet_ca([q_dim, h_dim * self.k], act="Swish", dropout=dropout)
        self.k_net = nn.Sequential(
            FCNet_ca([k_dim, h_dim * 2], act="GELU", norm="LayerNorm", residual=True),
            FCNet_ca([h_dim * 2, h_dim * self.k], act="GELU", dropout=dropout),
        )

        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        self.h_mat_vq = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.h_bias_vq = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        self.h_mat_vk = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.h_bias_vk = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        self.h_mat_qk = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.h_bias_qk = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())

        nn.init.xavier_normal_(self.h_mat_vq)
        nn.init.xavier_normal_(self.h_mat_vk)
        nn.init.xavier_normal_(self.h_mat_qk)
        nn.init.kaiming_uniform_(self.h_bias_vq, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.h_bias_vk, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.h_bias_qk, nonlinearity="linear")

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, lhs, rhs, att_map):
        fusion_logits = torch.einsum("bvk,bvq,bqk->bk", (lhs, att_map, rhs))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def _compute_att_maps(self, h_mat, h_bias, lhs, rhs, lhs_mask, rhs_mask, softmax, lhs_num, rhs_num):
        att_maps = torch.einsum("xhyk,bvk,bqk->bhvq", (h_mat, lhs, rhs)) + h_bias
        if softmax:
            att_maps = compute_attention_mask(lhs_mask, rhs_mask, att_maps, self.h_out, lhs_num, rhs_num)
        return att_maps

    def _sum_heads(self, lhs, rhs, att_maps):
        logits = self.attention_pooling(lhs, rhs, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits = logits + self.attention_pooling(lhs, rhs, att_maps[:, i, :, :])
        return logits

    def forward(self, v, q, k, v_mask=None, q_mask=None, k_mask=None, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        k_num = k.size(1)

        v_ = self.v_net(v)
        q_ = self.q_net(q)
        k_ = self.k_net(k)

        att_maps_vq = self._compute_att_maps(
            self.h_mat_vq, self.h_bias_vq, v_, q_, v_mask, q_mask, softmax, v_num, q_num
        )
        logits = self._sum_heads(v_, q_, att_maps_vq)

        att_maps_vk = None
        if self.fusion_method in {"vk", "bi_vk"}:
            att_maps_vk = self._compute_att_maps(
                self.h_mat_vk, self.h_bias_vk, v_, k_, v_mask, k_mask, softmax, v_num, k_num
            )
            logits = logits + self._sum_heads(v_, k_, att_maps_vk)

        att_maps_qk = None
        if self.fusion_method in {"qk", "bi_qk", "bi_vk"}:
            att_maps_qk = self._compute_att_maps(
                self.h_mat_qk, self.h_bias_qk, q_, k_, q_mask, k_mask, softmax, q_num, k_num
            )
            logits = logits + self._sum_heads(q_, k_, att_maps_qk)

        logits = self.bn(logits)
        return (
            logits,
            att_maps_vq[:, -1, :, :],
            None if att_maps_vk is None else att_maps_vk[:, -1, :, :],
            None if att_maps_qk is None else att_maps_qk[:, -1, :, :],
        )


class Seed42AceKBANLayer(nn.Module):
    """Compatibility layer for the seed_42 ACE prompt_cat checkpoints."""

    def __init__(self, v_dim, q_dim, h_dim, k_dim, h_out, act=nn.ReLU(), dropout=0.2, k=3, c=32, fusion_method="vk"):
        super().__init__()
        self.c = c
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.h_dim = h_dim
        self.h_out = h_out
        self.fusion_method = fusion_method

        self.v_net = FCNet_ca([v_dim, h_dim * self.k], act="GELU", norm="LayerNorm", residual=True)
        self.q_net = FCNet_ca([q_dim, h_dim * self.k], act="Swish", dropout=dropout)
        self.k_net = nn.Sequential(
            FCNet_ca([k_dim, h_dim * 2], act="GELU", norm="LayerNorm", residual=True),
            FCNet_ca([h_dim * 2, h_dim * self.k], act="GELU", dropout=dropout),
        )
        self.k_cross_attn = nn.MultiheadAttention(embed_dim=h_dim * self.k, num_heads=4)
        self.k_gate = nn.Sequential(nn.Linear(h_dim * self.k, 1))

        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        self.h_mat_vq = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.h_bias_vq = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        self.h_mat_vk = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.h_bias_vk = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        self.h_mat_qk = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.h_bias_qk = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())

        nn.init.xavier_normal_(self.h_mat_vq)
        nn.init.xavier_normal_(self.h_mat_vk)
        nn.init.xavier_normal_(self.h_mat_qk)
        nn.init.kaiming_uniform_(self.h_bias_vq, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.h_bias_vk, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.h_bias_qk, nonlinearity="linear")

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, lhs, rhs, att_map):
        fusion_logits = torch.einsum("bvk,bvq,bqk->bk", (lhs, att_map, rhs))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def _compute_att_maps(self, h_mat, h_bias, lhs, rhs, lhs_mask, rhs_mask, softmax, lhs_num, rhs_num):
        att_maps = torch.einsum("xhyk,bvk,bqk->bhvq", (h_mat, lhs, rhs)) + h_bias
        if softmax:
            att_maps = compute_attention_mask(lhs_mask, rhs_mask, att_maps, self.h_out, lhs_num, rhs_num)
        return att_maps

    def _sum_heads(self, lhs, rhs, att_maps):
        logits = self.attention_pooling(lhs, rhs, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits = logits + self.attention_pooling(lhs, rhs, att_maps[:, i, :, :])
        return logits

    def forward(self, v, q, k, v_mask=None, q_mask=None, k_mask=None, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        k_num = k.size(1)

        v_ = self.v_net(v)
        q_ = self.q_net(q)
        k_ = self.k_net(k)

        k_cross, _ = self.k_cross_attn(
            query=v_.transpose(0, 1),
            key=k_.transpose(0, 1),
            value=k_.transpose(0, 1),
        )
        k_cross = k_cross.transpose(0, 1)
        gate = torch.sigmoid(self.k_gate(k_cross))
        v_ = v_ + gate * k_cross

        att_maps_vq = self._compute_att_maps(
            self.h_mat_vq, self.h_bias_vq, v_, q_, v_mask, q_mask, softmax, v_num, q_num
        )
        logits = self._sum_heads(v_, q_, att_maps_vq)

        att_maps_vk = self._compute_att_maps(
            self.h_mat_vk, self.h_bias_vk, v_, k_, v_mask, k_mask, softmax, v_num, k_num
        )
        att_maps_qk = self._compute_att_maps(
            self.h_mat_qk, self.h_bias_qk, q_, k_, q_mask, k_mask, softmax, q_num, k_num
        )

        logits = self.bn(logits)
        return logits, att_maps_vq[:, -1, :, :], att_maps_vk[:, -1, :, :], att_maps_qk[:, -1, :, :]


def _clean_finetuned_state(finetuned_state):
    cleaned = {}
    for k, v in finetuned_state.items():
        k = k.replace("module.", "")
        if k.startswith("base_feature_extractor."):
            k = k.replace("base_feature_extractor.", "model.")
        if k.startswith("graph_feature_extractor."):
            k = k.replace("graph_feature_extractor.", "base_feature_extractor.")
        cleaned[k] = v
    return cleaned


def _is_legacy_ace_checkpoint(cleaned):
    predictor_w = cleaned.get("predictor.0.weight")
    prompt_proj_w = cleaned.get("prompt_fusion.prompt_projection_model.in_proj.weight")
    return (
        "bcn_list.h_mat_vk" in cleaned
        and predictor_w is not None
        and tuple(predictor_w.shape) == (256, 1152)
        and prompt_proj_w is not None
        and tuple(prompt_proj_w.shape) == (384, 3072)
    )


def _is_seed42_ace_prompt_cat_checkpoint(cleaned):
    predictor_w = cleaned.get("predictor.0.weight")
    prompt_proj_w = cleaned.get("prompt_projection_model.0.weight")
    prompt_linear_w = cleaned.get("prompt_linear_model.0.weight")
    return (
        "bcn_list.h_mat_vk" in cleaned
        and "bcn_list.k_cross_attn.in_proj_weight" in cleaned
        and predictor_w is not None
        and tuple(predictor_w.shape) == (384, 3072)
        and prompt_proj_w is not None
        and tuple(prompt_proj_w.shape) == (384, 6144)
        and prompt_linear_w is not None
        and tuple(prompt_linear_w.shape) == (384, 768)
    )


def _infer_n_tasks_from_checkpoint(cleaned):
    """Infer predictor output dimension from checkpoint for screening datasets."""
    out_bias = cleaned.get("predictor.3.bias")
    if out_bias is not None and out_bias.ndim == 1 and out_bias.shape[0] > 0:
        return int(out_bias.shape[0])
    out_weight = cleaned.get("predictor.3.weight")
    if out_weight is not None and out_weight.ndim == 2 and out_weight.shape[0] > 0:
        return int(out_weight.shape[0])
    out_bias = cleaned.get("predictor.bias")
    if out_bias is not None and out_bias.ndim == 1 and out_bias.shape[0] > 0:
        return int(out_bias.shape[0])
    out_weight = cleaned.get("predictor.weight")
    if out_weight is not None and out_weight.ndim == 2 and out_weight.shape[0] > 0:
        return int(out_weight.shape[0])
    return None


def build_text_dict(device: torch.device, text_model_name: str, dataset_base_path: str, train_text_model: bool):
    text_path = os.path.join(dataset_base_path, "text", "phar_question_howmany_gpt4o_27.json")
    with open(text_path, "r", encoding="utf-8") as fp:
        text_list = json.load(fp)

    select_question = []
    select_description = []
    phar_question_name = []
    for _, items in text_list.items():
        for item in items:
            select_question.append(f"Question: {item['question']}")
            select_description.append(f"Description: {item['description']}")
            phar_question_name.append(item["type"])

    text_model = TextEncoder(model_name=text_model_name, load=True)
    question_texts, question_masks = text_model.tokenize(select_question, max_length=96)
    description_texts, description_masks = text_model.tokenize(select_description, max_length=96)

    if question_texts.dim() == 1:
        question_texts = question_texts.unsqueeze(0)
        question_masks = question_masks.unsqueeze(0)
        description_texts = description_texts.unsqueeze(0)
        description_masks = description_masks.unsqueeze(0)

    if not train_text_model:
        with torch.no_grad():
            q_emb = text_model(input_ids=question_texts, attention_mask=question_masks)
            d_emb = text_model(input_ids=description_texts, attention_mask=description_masks)
        text_dict = {
            "question_text_embeddings": q_emb.to(device),
            "discription_text_embeddings": d_emb.to(device),
            "question_masks": question_masks.to(device),
            "discription_mask": description_masks.to(device),
            "phar_question_name": phar_question_name,
            "question_tokens": question_texts,
            "know_tokens": description_texts,
            "question_texts": select_question,
        }
    else:
        text_dict = {
            "question_masks": question_masks.to(device),
            "discription_mask": description_masks.to(device),
            "phar_question_name": phar_question_name,
            "question_tokens": question_texts.to(device),
            "know_tokens": description_texts.to(device),
            "question_texts": select_question,
        }

    return text_dict, text_model


def init_model_for_inference(
    device: torch.device,
    pretrained_base_path: str,
    finetuned_ckpt_path: str,
    dataset: PharmaQADataset,
    text_model_name: str,
    smiles_model_name: str,
    dropout: float,
    n_layers: int,
    projection_layers: int,
    train_text_model: bool,
    text_model: Optional[TextEncoder] = None,
):
    base_config = config_dict["base"]
    text_encoder_config = config_dict["text_encoder"][text_model_name]
    smile_encoder_config = config_dict["smiles_encoder"][smiles_model_name]
    finetuned_state = torch.load(finetuned_ckpt_path, map_location="cpu")
    cleaned = _clean_finetuned_state(finetuned_state)
    use_legacy_ace = _is_legacy_ace_checkpoint(cleaned)
    use_seed42_ace_prompt_cat = _is_seed42_ace_prompt_cat_checkpoint(cleaned)
    inferred_n_tasks = _infer_n_tasks_from_checkpoint(cleaned) or dataset.n_tasks

    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = LiGhT(
        d_node_feats=base_config["d_node_feats"],
        d_edge_feats=base_config["d_edge_feats"],
        d_g_feats=base_config["d_g_feats"],
        d_fp_feats=dataset.d_fps,
        d_md_feats=dataset.d_mds,
        d_hpath_ratio=base_config["d_hpath_ratio"],
        n_mol_layers=base_config["n_mol_layers"],
        path_length=base_config["path_length"],
        n_heads=base_config["n_heads"],
        n_ffn_dense_layers=base_config["n_ffn_dense_layers"],
        input_drop=0,
        attn_drop=dropout,
        feat_drop=dropout,
        n_node_types=vocab.vocab_size,
    ).to(device)

    # 先加载 base 预训练参数（和 finetune 时一致）
    base_state = torch.load(pretrained_base_path, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in base_state.items()})

    # finetune 阶段会删除这些 heads，这里保持一致
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor

    model.text_question_prompt_proj = MLP(
        text_encoder_config["out_dim"],
        text_encoder_config["out_dim"],
        projection_layers,
        nn.GELU(),
        d_hidden_feats=text_encoder_config["out_dim"],
        dropout=dropout,
    ).to(device)
    model.text_disp_prompt_proj = MLP(
        text_encoder_config["out_dim"],
        text_encoder_config["out_dim"],
        projection_layers,
        nn.GELU(),
        d_hidden_feats=text_encoder_config["out_dim"],
        dropout=dropout,
    ).to(device)
    model.smiles_embed_proj = MLP(
        smile_encoder_config["out_dim"],
        text_encoder_config["out_dim"],
        projection_layers,
        nn.GELU(),
        d_hidden_feats=text_encoder_config["out_dim"],
        dropout=dropout,
    ).to(device)

    if use_legacy_ace:
        ban_out_dim = int(cleaned["prompt_linear_model.0.weight"].shape[1])
        head_out_dim = int(cleaned["prompt_fusion.prompt_projection_model.out_proj.weight"].shape[0])
        predictor_input_dim = int(cleaned["predictor.0.weight"].shape[1])
        model.bcn_list = LegacyAceKBANLayer(
            v_dim=base_config["d_g_feats"],
            q_dim=text_encoder_config["out_dim"],
            k_dim=text_encoder_config["out_dim"],
            h_dim=ban_out_dim,
            h_out=2,
            dropout=dropout,
            act=nn.GELU(),
            k=3,
            fusion_method="vk",
        ).to(device)
        model.prompt_fusion = TwoStagePromptFusion(
            dim=ban_out_dim,
            dim_out=head_out_dim,
            phar_num_list=phar_num_list,
            dropout=dropout,
            projection_layers=projection_layers,
            d_hidden_feats=head_out_dim,
        ).to(device)
        model.prompt_linear_model = get_predictor(
            d_input_feats=ban_out_dim,
            n_tasks=1,
            n_layers=n_layers,
            predictor_drop=dropout,
            device=device,
            d_hidden_feats=256,
        )
        model.predictor = get_predictor(
            d_input_feats=predictor_input_dim,
            n_tasks=inferred_n_tasks,
            n_layers=n_layers,
            predictor_drop=dropout,
            device=device,
            d_hidden_feats=256,
        )
        # Legacy ACE checkpoints concatenate prompt fusion with graph readout only.
        model.legacy_ace_graph_readout_only = True
    elif use_seed42_ace_prompt_cat:
        ban_out_dim = int(cleaned["prompt_linear_model.0.weight"].shape[1])
        pharma_hidden_dim = int(cleaned["pharma_projection_model_list.0.0.weight"].shape[0])
        pharma_output_dim = int(cleaned["pharma_projection_model_list.0.3.weight"].shape[0])
        prompt_hidden_dim = int(cleaned["prompt_projection_model.0.weight"].shape[0])
        prompt_output_dim = int(cleaned["prompt_projection_model.3.weight"].shape[0])
        predictor_hidden_dim = int(cleaned["predictor.0.weight"].shape[0])
        predictor_input_dim = int(cleaned["predictor.0.weight"].shape[1])
        model.bcn_list = Seed42AceKBANLayer(
            v_dim=base_config["d_g_feats"],
            q_dim=text_encoder_config["out_dim"],
            k_dim=text_encoder_config["out_dim"],
            h_dim=ban_out_dim,
            h_out=2,
            dropout=dropout,
            act=nn.GELU(),
            k=3,
            fusion_method="vk",
        ).to(device)
        model.prompt_linear_model = get_predictor(
            d_input_feats=ban_out_dim,
            n_tasks=1,
            n_layers=2,
            predictor_drop=dropout,
            device=device,
            d_hidden_feats=predictor_hidden_dim,
        )
        model.pharma_projection_model_list = nn.ModuleList(
            [
                get_predictor(
                    d_input_feats=phar_num_list[i] * ban_out_dim,
                    n_tasks=pharma_output_dim,
                    n_layers=2,
                    predictor_drop=dropout,
                    device=device,
                    d_hidden_feats=pharma_hidden_dim,
                )
                for i in range(len(phar_num_list))
            ]
        )
        model.prompt_projection_model = get_predictor(
            d_input_feats=len(phar_num_list) * pharma_output_dim,
            n_tasks=prompt_output_dim,
            n_layers=2,
            predictor_drop=dropout,
            device=device,
            d_hidden_feats=prompt_hidden_dim,
        )
        model.predictor = get_predictor(
            d_input_feats=predictor_input_dim,
            n_tasks=inferred_n_tasks,
            n_layers=2,
            predictor_drop=dropout,
            device=device,
            d_hidden_feats=predictor_hidden_dim,
        )
        model.base_prompt_proj = MLP(
            base_config["d_g_feats"],
            base_config["d_g_feats"],
            2,
            nn.GELU(),
            d_hidden_feats=base_config["d_g_feats"],
            dropout=dropout,
        ).to(device)
        model.graph_prompt_proj = MLP(
            base_config["d_g_feats"],
            base_config["d_g_feats"],
            2,
            nn.GELU(),
            d_hidden_feats=base_config["d_g_feats"],
            dropout=dropout,
        ).to(device)
        model.question_graph_feature_extractor = deepcopy(model.model)
        model.legacy_seed42_ace_prompt_projection = True
        model.legacy_seed42_ace_use_group_mlps = True
    else:
        model.bcn_list = KBANLayer(
            v_dim=base_config["d_g_feats"],
            q_dim=text_encoder_config["out_dim"],
            k_dim=text_encoder_config["out_dim"],
            h_dim=config_dict["BAN"]["out_dim"],
            h_out=2,
            dropout=dropout,
            act=nn.GELU(),
            k=3,
        ).to(device)
        model.prompt_fusion = TwoStagePromptFusion(
            dim=config_dict["BAN"]["out_dim"],
            dim_out=config_dict["head"]["out_dim"],
            phar_num_list=phar_num_list,
            dropout=dropout,
            projection_layers=projection_layers,
            d_hidden_feats=config_dict["head"]["out_dim"],
        ).to(device)
        model.prompt_linear_model = get_predictor(
            d_input_feats=config_dict["BAN"]["out_dim"],
            n_tasks=1,
            n_layers=n_layers,
            predictor_drop=dropout,
            device=device,
            d_hidden_feats=256,
        )
        model.predictor = get_predictor(
            d_input_feats=config_dict["head"]["out_dim"] + 3 * base_config["d_g_feats"],
            n_tasks=inferred_n_tasks,
            n_layers=n_layers,
            predictor_drop=dropout,
            device=device,
            d_hidden_feats=256,
        )

    if train_text_model and text_model is not None:
        model.text_model = text_model.to(device)

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def filter_valid_smiles(df: pd.DataFrame, smiles_col: str):
    valid_mask = []
    for smi in df[smiles_col].astype(str).tolist():
        mol = Chem.MolFromSmiles(smi)
        valid_mask.append(mol is not None)
    return df.loc[valid_mask].copy()


def ensure_screening_dataset(
    source_csv_path: str,
    source_smiles_col: str,
    dataset_base_path: str,
    cache_dataset_name: str,
    overwrite: bool = False,
):
    """
    创建一个模型可读的数据集目录：
    datasets_base/cache_dataset_name/cache_dataset_name.csv 仅含 smiles + label(0)
    同时保存 meta.csv 方便把预测结果 merge 回来
    """
    out_dir = os.path.join(dataset_base_path, cache_dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    model_csv_path = os.path.join(out_dir, f"{cache_dataset_name}.csv")
    meta_csv_path = os.path.join(out_dir, f"{cache_dataset_name}.meta.csv")

    if (not overwrite) and os.path.exists(model_csv_path) and os.path.exists(meta_csv_path):
        return model_csv_path, meta_csv_path

    df = pd.read_csv(source_csv_path)
    if source_smiles_col not in df.columns:
        raise ValueError(f"输入文件缺少列 {source_smiles_col}，实际列为: {list(df.columns)[:30]} ...")

    df = df.dropna(subset=[source_smiles_col]).copy()
    df = filter_valid_smiles(df, source_smiles_col)

    meta_df = df.copy()
    meta_df.rename(columns={source_smiles_col: "SMILES"}, inplace=True)
    meta_df.to_csv(meta_csv_path, index=False)

    model_df = pd.DataFrame({"smiles": meta_df["SMILES"].astype(str), "label": np.zeros(len(meta_df), dtype=float)})
    model_df.to_csv(model_csv_path, index=False)

    return model_csv_path, meta_csv_path


def maybe_run_preprocess(dataset_base_path: str, cache_dataset_name: str, path_length: int, num_workers: int):
    """
    如果缺少必要特征文件，则自动运行 preprocess 脚本生成：
    - {dataset}_{path_length}.pkl
    - rdkfp1-7_512.npz
    - molecular_descriptors.npz
    - phar_features_lmdb/
    - smiles_embeddings_chembert_lmdb/
    """
    ds_dir = os.path.join(dataset_base_path, cache_dataset_name)
    need = {
        "graph_pkl": os.path.join(ds_dir, f"{cache_dataset_name}_{path_length}.pkl"),
        "fp": os.path.join(ds_dir, "rdkfp1-7_512.npz"),
        "md": os.path.join(ds_dir, "molecular_descriptors.npz"),
        "phar_lmdb": os.path.join(ds_dir, "phar_features_lmdb"),
        "smi_lmdb": os.path.join(ds_dir, "smiles_embeddings_chembert_lmdb"),
    }

    missing = []
    for k, p in need.items():
        if k.endswith("_lmdb"):
            if not os.path.isdir(p):
                missing.append(k)
        else:
            if not os.path.exists(p):
                missing.append(k)

    if not missing:
        return

    # 图/指纹/描述符
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(BASE_PATH, "scripts", "preprocess_dataset_graph.py"),
            "--data_path",
            dataset_base_path,
            "--dataset",
            cache_dataset_name,
            "--path_length",
            str(path_length),
            "--n_jobs",
            str(num_workers),
        ],
        cwd=BASE_PATH,
    )

    # 药效团特征
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(BASE_PATH, "scripts", "preprocess_dataset_phar.py"),
            "--dataset_base_path",
            dataset_base_path,
            "--dataset",
            cache_dataset_name,
            "--input",
            "phar",
            "--multiprocessing",
            "False",
            "--num_workers",
            str(max(1, min(8, num_workers))),
        ],
        cwd=BASE_PATH,
    )

    # SMILES embedding（chembert）
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(BASE_PATH, "scripts", "preprocess_dataset_phar.py"),
            "--dataset_base_path",
            dataset_base_path,
            "--dataset",
            cache_dataset_name,
            "--input",
            "smiles_embedding",
            "--multiprocessing",
            "False",
            "--num_workers",
            str(max(1, min(8, num_workers))),
        ],
        cwd=BASE_PATH,
    )


def mean_std_from_training_split(train_dataset_base_path: str, train_dataset_name: str, split_name: str, path_length: int):
    split_path = os.path.join(train_dataset_base_path, train_dataset_name, "splits", f"{split_name}.npy")
    graph_path = os.path.join(train_dataset_base_path, train_dataset_name, f"{train_dataset_name}_{path_length}.pkl")
    if (not os.path.exists(split_path)) or (not os.path.exists(graph_path)):
        raise FileNotFoundError(f"找不到训练集 split 或 graph：{split_path} 或 {graph_path}")

    train_idx = np.load(split_path, allow_pickle=True)[0]
    _, label_dict = load_graphs(graph_path)
    labels = label_dict["labels"][train_idx]
    mean = torch.from_numpy(np.nanmean(labels.numpy(), axis=0)).float()
    std = torch.from_numpy(np.nanstd(labels.numpy(), axis=0)).float()
    return mean, std


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    text_dict: dict,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
):
    all_smiles = []
    all_pred = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            (
                smiles,
                graphs,
                fps,
                mds,
                labels,
                phar_targets,
                phar_target_mx,
                atom_phar_target_map,
                smiles_embed,
                smiles_mask,
            ) = batch

            graphs = graphs.to(device)
            fps = fps.to(device)
            mds = mds.to(device)
            smiles_embed = smiles_embed.to(device)
            smiles_mask = smiles_mask.to(device)

            pred, _, _ = model.forward_pharmaPrompt(
                graphs, fps, mds, text=text_dict, smiles_embed=smiles_embed, smiles_mask=smiles_mask
            )

            pred = pred.detach().cpu()
            if (mean is not None) and (std is not None):
                pred = pred * std + mean

            all_smiles.extend(list(smiles))
            all_pred.extend(pred.numpy().reshape(-1).tolist())

    return pd.DataFrame({"SMILES": all_smiles, "prediction": all_pred})


def parse_args():
    p = argparse.ArgumentParser(description="Use finetuned HPK1_IC50 checkpoint to score DrugBank and export top20")
    p.add_argument("--checkpoint_dir", type=str, required=True, help="目录内应包含 best_model.pth")
    p.add_argument("--checkpoint_name", type=str, default="best_model.pth")

    p.add_argument("--source_drugbank_csv", type=str, required=True)
    p.add_argument("--source_smiles_col", type=str, default="SMILES")

    # 缓存数据集必须放在 datasets/vs 下，这样 preprocess_dataset_phar.py 会走 vs 分支
    p.add_argument("--dataset_base_path", type=str, default=os.path.join(BASE_PATH, "datasets", "vs"))
    p.add_argument("--cache_dataset_name", type=str, default="DrugBank_cache")
    p.add_argument("--overwrite_cache_csv", action="store_true")

    p.add_argument("--pretrained_base_path", type=str, default=os.path.join(BASE_PATH, "pretrained", "base", "base.pth"))

    # 训练集统计量（mean/std）来自 ligand 数据集目录
    p.add_argument("--train_dataset_base_path", type=str, default=os.path.join(BASE_PATH, "datasets", "ligand"))
    p.add_argument("--train_dataset_name", type=str, default="HPK1_IC50")
    p.add_argument("--train_split_name", type=str, default="scaffold-3")
    p.add_argument("--path_length", type=int, default=5)
    p.add_argument("--use_norm_reg", action="store_true")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=16, help="预处理 graph 用的 n_jobs")

    p.add_argument("--text_model_name", type=str, default="pubmed")
    p.add_argument("--smiles_model_name", type=str, default="chembert")
    p.add_argument("--train_text_model", action="store_true")

    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--projection_layers", type=int, default=2)

    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    if not os.path.exists(args.pretrained_base_path):
        raise FileNotFoundError(f"找不到 base 预训练权重: {args.pretrained_base_path}")

    # 1) 构建 screening 数据集（只含 smiles+label）
    model_csv_path, meta_csv_path = ensure_screening_dataset(
        source_csv_path=args.source_drugbank_csv,
        source_smiles_col=args.source_smiles_col,
        dataset_base_path=args.dataset_base_path,
        cache_dataset_name=args.cache_dataset_name,
        overwrite=args.overwrite_cache_csv,
    )

    # 2) 预处理生成图/特征（缺啥补啥）
    maybe_run_preprocess(
        dataset_base_path=args.dataset_base_path,
        cache_dataset_name=args.cache_dataset_name,
        path_length=args.path_length,
        num_workers=args.num_workers,
    )

    # 3) 加载 dataset + collator
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    collator = Collator_pharmaPrompt(max_length=128)
    ds = PharmaQADataset(
        root_path=args.dataset_base_path,
        dataset=args.cache_dataset_name,
        dataset_type="regression",
        path_length=args.path_length,
        split_name="scaffold-0",
        split=None,
        train_kpgt="False",
        text_max_len=128,
        phar_question_name=[],
        use_norm_reg=False,
        smi_encoder_name=args.smiles_model_name,
        phar_load_method="lmdb",
        smiles_load_method="lmdb",
        split_method="kpgt",
        base_model_encoder="LiGhT",
        ace_clip_test=False,
        prompt_graph_feature_extractor="LiGhT",
        seed=42,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collator)

    # 4) 文本问题 embedding
    text_dict, text_model = build_text_dict(
        device=device,
        text_model_name=args.text_model_name,
        dataset_base_path=os.path.dirname(args.dataset_base_path),
        train_text_model=args.train_text_model,
    )

    # 5) 初始化模型并加载 checkpoint
    model = init_model_for_inference(
        device=device,
        pretrained_base_path=args.pretrained_base_path,
        finetuned_ckpt_path=ckpt_path,
        dataset=ds,
        text_model_name=args.text_model_name,
        smiles_model_name=args.smiles_model_name,
        dropout=args.dropout,
        n_layers=args.n_layers,
        projection_layers=args.projection_layers,
        train_text_model=args.train_text_model,
        text_model=text_model,
    )

    # 6) 反标准化（可选）
    mean = std = None
    if args.use_norm_reg:
        mean, std = mean_std_from_training_split(
            train_dataset_base_path=args.train_dataset_base_path,
            train_dataset_name=args.train_dataset_name,
            split_name=args.train_split_name,
            path_length=args.path_length,
        )

    # 7) 预测 + topk
    pred_df = predict(model, loader, device, text_dict, mean, std)
    pred_df = pred_df.sort_values("prediction", ascending=False).reset_index(drop=True)
    top_df = pred_df.head(args.topk).copy()

    # merge 回 meta（包含 DrugBank ID / Name 等）
    meta_df = pd.read_csv(meta_csv_path)
    merged = meta_df.merge(top_df, on="SMILES", how="right")
    merged = merged.sort_values("prediction", ascending=False).reset_index(drop=True)

    # 输出
    out_dir = args.output_dir.strip() or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    full_out = os.path.join(out_dir, f"{args.cache_dataset_name}_predictions_full.csv")
    top_out = os.path.join(out_dir, f"{args.cache_dataset_name}_top{args.topk}.csv")
    pred_df.to_csv(full_out, index=False)
    merged.to_csv(top_out, index=False)
    print(f"[OK] full predictions -> {full_out}")
    print(f"[OK] top{args.topk} -> {top_out}")


if __name__ == "__main__":
    main()

