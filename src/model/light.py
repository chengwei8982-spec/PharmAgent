from cProfile import label
from distutils import text_file
import json
import os
import sys

# from pydantic import BaseModel
import torch
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import numpy as np
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence

base_path =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.chembert.tokenizer import ChemBertTokenizer
from src.model.smi_model import ChemBert

from src.data.featurizer import VIRTUAL_ATOM_FEATURE_PLACEHOLDER, VIRTUAL_BOND_FEATURE_PLACEHOLDER

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Residual(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, activation, d_hidden_feats=d_out_feats * 4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        return x


class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers - 2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats


class TripletTransformer(nn.Module):
    def __init__(self,
                 d_feats,
                 d_hpath_ratio,
                 path_length,
                 n_heads,
                 n_ffn_dense_layers,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU()):
        super(TripletTransformer, self).__init__()
        self.d_feats = d_feats
        self.d_trip_path = d_feats // d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop, activation)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}

    def forward(self, g, triplet_h, dist_attn, path_attn):
        g = g.local_var()
        new_triplet_h = self.attention_norm(triplet_h)
        qkv = self.qkv(new_triplet_h).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']), -1, 1) + path_attn.reshape(
            len(g.edata['node_attn']), -1, 1)
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))

        g.ndata['hv'] = v.view(-1, self.d_feats)
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats // self.n_heads)) * g.edata['sa']).view(-1,
                                                                                                                    self.d_feats)

        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))
        return self.node_out_layer(triplet_h, g.ndata['agg_h'])

    def _device(self):
        return next(self.parameters()).device


class LiGhT(nn.Module):
    def __init__(self,
                 d_g_feats,
                 d_hpath_ratio,
                 path_length,
                 n_mol_layers=2,
                 n_heads=4,
                 n_ffn_dense_layers=4,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU()):
        super(LiGhT, self).__init__()
        self.n_mol_layers = n_mol_layers
        self.n_heads = n_heads
        self.path_length = path_length
        self.d_g_feats = d_g_feats
        self.d_trip_path = d_g_feats // d_hpath_ratio

        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Distance Attention
        self.path_len_emb = nn.Embedding(path_length + 1, d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, d_g_feats)
        self.self_loop_emb = nn.Embedding(1, d_g_feats)
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(self.d_g_feats, self.d_g_feats),
            activation,
            nn.Linear(self.d_g_feats, n_heads)
        )
        # Path Attention  
        self.trip_fortrans = nn.ModuleList([
            MLP(d_g_feats, self.d_trip_path, 2, activation) for _ in range(self.path_length)
        ])
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path),
            activation,
            nn.Linear(self.d_trip_path, n_heads)
        )
        # Molecule Transformer Layers
        self.mol_T_layers = nn.ModuleList([
            TripletTransformer(d_g_feats, d_hpath_ratio, path_length, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
                               activation) for _ in range(n_mol_layers)
        ])

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def _featurize_path(self, g, path_indices):
        mask = (path_indices[:, :] >= 0).to(torch.int32)
        path_feats = torch.sum(mask, dim=-1)
        path_feats = self.path_len_emb(path_feats)
        path_feats[g.edata['vp'] == 1] = self.virtual_path_emb.weight  # virtual path
        path_feats[g.edata['sl'] == 1] = self.self_loop_emb.weight  # self loop
        return path_feats

    def _init_path(self, g, triplet_h, path_indices):
        g = g.local_var()
        path_indices[path_indices < -99] = -1
        path_h = []
        for i in range(self.path_length):
            path_h.append(torch.cat(
                [self.trip_fortrans[i](triplet_h), torch.zeros(size=(1, self.d_trip_path)).to(self._device())], dim=0)[
                              path_indices[:, i]])
        path_h = torch.stack(path_h, dim=-1)
        mask = (path_indices >= 0).to(torch.int32)
        path_size = torch.sum(mask, dim=-1, keepdim=True)
        path_h = torch.sum(path_h, dim=-1) / path_size
        return path_h

    def forward(self, g, triplet_h):
        path_indices = g.edata['path']
        dist_h = self._featurize_path(g, path_indices)
        path_h = self._init_path(g, triplet_h, path_indices)
        dist_attn, path_attn = self.dist_attn_layer(dist_h), self.path_attn_layer(path_h)
        for i in range(self.n_mol_layers):
            triplet_h = self.mol_T_layers[i](g, triplet_h, dist_attn, path_attn)
        return triplet_h

    def _device(self):
        return next(self.parameters()).device


class AtomEmbedding(nn.Module):
    def __init__(
            self,
            d_atom_feats,
            d_g_feats,
            input_drop):
        super(AtomEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, pair_node_feats, indicators):
        pair_node_h = self.in_proj(pair_node_feats)
        # Ensure the virtual_atom_emb weights have the same dtype as pair_node_h
        virtual_weight = self.virtual_atom_emb.weight.to(dtype=pair_node_h.dtype)
        pair_node_h[indicators == VIRTUAL_ATOM_FEATURE_PLACEHOLDER, 1, :] = virtual_weight
        return torch.sum(self.input_dropout(pair_node_h), dim=-2)


class BondEmbedding(nn.Module):
    def __init__(
            self,
            d_bond_feats,
            d_g_feats,
            input_drop):
        super(BondEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_bond_feats, d_g_feats)
        self.virutal_bond_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, edge_feats, indicators):
        edge_h = self.in_proj(edge_feats)
        # Ensure the virutal_bond_emb weights have the same dtype as edge_h
        virtual_weight = self.virutal_bond_emb.weight.to(dtype=edge_h.dtype)
        edge_h[indicators == VIRTUAL_BOND_FEATURE_PLACEHOLDER] = virtual_weight
        return self.input_dropout(edge_h)


class TripletEmbedding(nn.Module):
    def __init__(
            self,
            d_g_feats,
            d_fp_feats,
            d_md_feats,
            activation=nn.GELU()):
        super(TripletEmbedding, self).__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)

    def forward(self, node_h, edge_h, fp, md, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        triplet_h[indicators == 1] = self.fp_proj(fp)
        triplet_h[indicators == 2] = self.md_proj(md)
        return triplet_h


class LiGhTPredictor(nn.Module):
    def __init__(self,
                 d_node_feats=40,
                 d_edge_feats=12,
                 d_g_feats=128,
                 d_fp_feats=512,
                 d_md_feats=200,
                 d_hpath_ratio=1,
                 n_mol_layers=2,
                 path_length=5,
                 n_heads=4,
                 n_ffn_dense_layers=2,
                 input_drop=0.,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU(),
                 n_node_types=1,
                 readout_mode='mean'
                 ):
        super(LiGhTPredictor, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Model
        self.model = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )

        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, n_node_types)
        )
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_md_feats)
        )

        self.apply(lambda module: init_params(module))

    def forward(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h[g.ndata['mask'] == 1] = self.mask_emb.weight
        # Model
        triplet_h = self.model(g, triplet_h)
        # Predict
        return self.node_predictor(triplet_h[g.ndata['mask'] >= 1]), self.fp_predictor(
            triplet_h[indicators == 1]), self.md_predictor(triplet_h[indicators == 2])

    def forward_tune(self, g, fp, md, text=None, text_mask=None, mode='downstream'):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return self.predictor(g_feats)
    
    def forward_pharmagent(self, g, fp, md, text=None, smiles_embed=None,smiles_mask=None):
        
        ##################### get graph feature
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        graph_other_feats = torch.stack([fp_vn, md_vn], dim=1)
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        ##################### get graph feature

        ##################### get node feature
        num_nodes_per_graph = g.batch_num_nodes()
        node_feat_seqs = torch.split(triplet_h[indicators <= 0], num_nodes_per_graph.tolist())
        node_feats = pad_sequence(node_feat_seqs, batch_first=True)
        node_mask = (torch.arange(node_feats.size(1)).to(node_feats.device)[None, :] < torch.tensor(num_nodes_per_graph)[:, None]).float()
        
        node_feats = torch.cat([graph_other_feats, node_feats], dim=1) 
        node_mask = torch.cat([torch.ones_like(graph_other_feats[:,:,0]), node_mask], dim=1)
        # node_feats = self.node_prompt_proj(node_feats)
        # graph_other_feats = self.graph_other_prompt_proj(graph_other_feats)
        ##################### get node feature

        ##################### get text feature
        text_question_embeddings = text['question_text_embeddings']
        text_question_mask = text['question_masks']

        text_disp_embeddings = text['discription_text_embeddings']
        text_disp_mask = text['discription_mask']
        ##################### get text feature

        smiles_embeddings = smiles_embed
        smiles_mask = smiles_mask
        ##################### BAN network
        logits_list, att_vq_list, att_vk_list, att_qk_list = [], [], [], []
        batch_size = node_feats.size()[0]

        text_question_reprs = self.text_question_prompt_proj(text_question_embeddings)
        text_disp_reprs = self.text_disp_prompt_proj(text_disp_embeddings)
        smiles_embed_reprs = self.smiles_embed_proj(smiles_embeddings)

        batch_text_question_reprs = text_question_reprs.unsqueeze(1).repeat(1, batch_size, 1, 1)
        batch_text_disp_reprs = text_disp_reprs.unsqueeze(1).repeat(1, batch_size, 1, 1)
        batch_text_question_mask = text_question_mask.unsqueeze(1).repeat(1, batch_size, 1).float()
        batch_text_disp_mask = text_disp_mask.unsqueeze(1).repeat(1, batch_size, 1).float()

        for idx in range(len(text_question_embeddings)):
            kno_question_out = batch_text_question_reprs[idx]
            kno_question_out_mask = batch_text_question_mask[idx]

            kno_dis_out = batch_text_disp_reprs[idx]
            kno_dis_out_mask = batch_text_disp_mask[idx]

            kno_dis_out = torch.cat((smiles_embed_reprs, kno_dis_out), dim=1)
            kno_dis_out_mask = torch.cat((smiles_mask.float(), kno_dis_out_mask), dim=1)
                
            answer_out, atten_vq, atten_vk, atten_qk = self.bcn_list(node_feats, kno_question_out, kno_dis_out,
                                                                     node_mask, kno_question_out_mask,
                                                                     kno_dis_out_mask, softmax=True)
            logits_list.append(answer_out)
            att_vq_list.append(atten_vq)
            att_vk_list.append(atten_vk)
            att_qk_list.append(atten_qk)

        prompt_feat = torch.stack(logits_list, dim=1)
        
        ##################### get phar prompt feature
        molecules_prompt = self.prompt_fusion(prompt_feat)

        ############ get output feature
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        readout = torch.cat([fp_vn, md_vn, readout], dim=-1)
        # readout = self.base_feat_fusion(readout)
        g_feats = torch.cat([molecules_prompt, readout], dim=-1)
        pred = self.predictor(g_feats)

        ############ get phar output feature
        pred_phar_num = self.prompt_linear_model(prompt_feat).squeeze(-1)
  
        atten = torch.stack(att_vq_list, dim=2).sum(dim=-1)  # + torch.stack(att_vk_list, dim=2).sum(dim=-1)

        return pred, pred_phar_num, atten[:,2:,:]

    # Keep the legacy method name for server and existing script compatibility.
    def _forward_legacy_prompt(self, g, fp, md, text=None, smiles_embed=None, smiles_mask=None):
        return self.forward_pharmagent(g, fp, md, text=text, smiles_embed=smiles_embed, smiles_mask=smiles_mask)

    def forward_pharmaQA_BindingDB(self, g, fp, md, text=None, smiles_embed=None,smiles_mask=None):
        
        ##################### get graph feature
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        graph_other_feats = torch.stack([fp_vn, md_vn], dim=1)
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        ##################### get graph feature

        ##################### get node feature
        num_nodes_per_graph = g.batch_num_nodes()
        node_feat_seqs = torch.split(triplet_h[indicators <= 0], num_nodes_per_graph.tolist())
        node_feats = pad_sequence(node_feat_seqs, batch_first=True)
        node_mask = (torch.arange(node_feats.size(1)).to(node_feats.device)[None, :] < torch.tensor(num_nodes_per_graph)[:, None]).float()
        
        node_feats = torch.cat([graph_other_feats, node_feats], dim=1) 
        node_mask = torch.cat([torch.ones_like(graph_other_feats[:,:,0]), node_mask], dim=1)
        # node_feats = self.node_prompt_proj(node_feats)
        # graph_other_feats = self.graph_other_prompt_proj(graph_other_feats)
        ##################### get node feature

        ##################### get text feature
        text_question_embeddings = text['question_text_embeddings']
        text_question_mask = text['question_masks']

        text_disp_embeddings = text['discription_text_embeddings']
        text_disp_mask = text['discription_mask']
        ##################### get text feature

        smiles_embeddings = smiles_embed
        smiles_mask = smiles_mask
        ##################### BAN network
        logits_list, att_vq_list, att_vk_list, att_qk_list = [], [], [], []
        batch_size = node_feats.size()[0]

        text_question_reprs = self.text_question_prompt_proj(text_question_embeddings)
        text_disp_reprs = self.text_disp_prompt_proj(text_disp_embeddings)
        smiles_embed_reprs = self.smiles_embed_proj(smiles_embeddings)

        batch_text_question_reprs = text_question_reprs.unsqueeze(1).repeat(1, batch_size, 1, 1)
        batch_text_disp_reprs = text_disp_reprs.unsqueeze(1).repeat(1, batch_size, 1, 1)
        batch_text_question_mask = text_question_mask.unsqueeze(1).repeat(1, batch_size, 1).float()
        batch_text_disp_mask = text_disp_mask.unsqueeze(1).repeat(1, batch_size, 1).float()

        for idx in range(len(text_question_embeddings)):
            kno_question_out = batch_text_question_reprs[idx]
            kno_question_out_mask = batch_text_question_mask[idx]

            kno_dis_out = batch_text_disp_reprs[idx]
            kno_dis_out_mask = batch_text_disp_mask[idx]

            kno_dis_out = torch.cat((smiles_embed_reprs, kno_dis_out), dim=1)
            kno_dis_out_mask = torch.cat((smiles_mask.float(), kno_dis_out_mask), dim=1)
                
            answer_out, atten_vq, atten_vk, atten_qk = self.bcn_list(node_feats, kno_question_out, kno_dis_out,
                                                                     node_mask, kno_question_out_mask,
                                                                     kno_dis_out_mask, softmax=True)
            logits_list.append(answer_out)
            att_vq_list.append(atten_vq)
            att_vk_list.append(atten_vk)
            att_qk_list.append(atten_qk)

        prompt_feat = torch.stack(logits_list, dim=1)
        
        ##################### get phar prompt feature
        molecules_prompt = self.prompt_fusion(prompt_feat)

        ############ get output feature
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        base_graph_feat = torch.cat([fp_vn, md_vn, readout], dim=-1)
        graph_feat_output = self.base_feat_fusion(base_graph_feat)
        g_feats = torch.cat([molecules_prompt, graph_feat_output], dim=-1)

        ############ get phar output feature
        pred_phar_num = self.prompt_linear_model(prompt_feat).squeeze(-1)
  
        atten = torch.stack(att_vq_list, dim=2).sum(dim=-1)  # + torch.stack(att_vk_list, dim=2).sum(dim=-1)

        return g_feats, pred_phar_num, atten[:,2:,:]

    def generate_fps(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.ndata['ht'] = triplet_h
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return g_feats

    def get_feat_mat(self, g, fp, md):
        # get question graph feature
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        triplet_h = self.model(g, triplet_h)

        # Readout
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        molecule_node_repr = pad_sequence(torch.split(triplet_h[indicators <= 0], g.batch_num_nodes().tolist()),
                                          batch_first=False, padding_value=-999)
        vv2 = molecule_node_repr.new_ones(
            (max(g.batch_num_nodes().tolist()), g.batch_size, molecule_node_repr.shape[2])) * -999
        vv2[:molecule_node_repr.shape[0], :, :] = molecule_node_repr
        molecule_node_repr = vv2.transpose(0, 1)
        molecule_node_mask = (molecule_node_repr[:, :, 0] != -999).float()

        return molecule_node_repr, molecule_node_mask

    def get_know_graph_feat(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Model
        triplet_h = self.model(g, triplet_h)

        # Readout
        g.ndata['ht'] = triplet_h

        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]

        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return g_feats

    def get_feat_embedding(self, g, fp, md):
        # get question graph feature
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        return triplet_h


setattr(LiGhTPredictor, "forward_" + "pharma" + "Prompt", LiGhTPredictor._forward_legacy_prompt)


from transformers import BertModel, BertConfig, WordpieceTokenizer, BertTokenizer
from transformers import AutoTokenizer, AutoModel


class MolT5EmbeddingModule(torch.nn.Module):
    def __init__(self, model_path):
        super(MolT5EmbeddingModule, self).__init__()
        self.path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path)


class TextEncoder(nn.Module):
    def __init__(self, model_name, load=True,eval=False):
        super(TextEncoder, self).__init__()
        self.model_name = model_name
        self.tokenizer, self.model = self.load_model(model_name, load)
        self.dropout = nn.Dropout(0.1)
        if eval:
            self.model.requires_grad_(False)
        else:
            self.model.requires_grad_(True)
    def load_model(self, model_name, load):
        if model_name == 'scibert':
            tokenizer = AutoTokenizer.from_pretrained(
                f'{root_path}/pretrained/scibert_scivocab_uncased')
            model = BertModel(BertConfig(vocab_size=31090))
            if load:
                self.load_pretrained_model(model,
                                           f'{root_path}/checkpoints/MoleculeSTM_checkpoints/pretrained_model/text_model.pth')
        elif model_name == 'pubmed':
            tokenizer = AutoTokenizer.from_pretrained(
                f'{root_path}/pretrained/BiomedBERT')
            model = BertModel(BertConfig.from_pretrained(
                f'{root_path}/pretrained/BiomedBERT'))
            if load:
                self.load_pubmed_model(model,
                                       f'{root_path}/checkpoints/DEN1/pytorch_model.bin')
        elif model_name == 'chembert':
            with open(f'{root_path}/src/model/chembert/config.json', 'r') as f:
                config = json.load(f)
            tokenizer = ChemBertTokenizer(
                vocab_path=f'{root_path}/src/model/chembert/vocab.json')
            model = ChemBert(**config)
            if load:
                self.load_chembert_model(model,
                                         f'{root_path}/checkpoints/DEN1/pytorch_model.bin')
        elif model_name == 'molT5':
            path = '../pretrained/molT5/'
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModel.from_pretrained(path)

        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        return tokenizer, model

    def load_pretrained_model(self, model, model_path):
        print(f"Loading from pretrained model {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

    def load_pubmed_model(self, model, checkpoint_path):
        print(f"Loading PubMed model from {checkpoint_path}...")
        ckpt_all = torch.load(checkpoint_path, map_location='cpu')
        ckpt = {key[len('model.den.txt_encoder.'):]: value for key, value in ckpt_all.items() if
                key.startswith('model.den.txt_encoder.')}
        model.pooler = None
        model.load_state_dict(ckpt, strict=False)

    def load_chembert_model(self, model, checkpoint_path):
        print(f"Loading ChemBert model from {checkpoint_path}...")
        ckpt_all = torch.load(checkpoint_path, map_location='cpu')
        ckpt = {key[len('model.den.smi_encoder.'):]: value for key, value in ckpt_all.items() if
                key.startswith('model.den.smi_encoder.')}
        model.load_state_dict(ckpt)

    def tokenize(self, text, max_length):
        if self.model_name == 'chembert':
            # ChemBertTokenizer doesn't have __call__ method, use encode instead
            # For ChemBertTokenizer, we need to handle individual text items
            if isinstance(text, list):
                all_ids = []
                all_masks = []
                for t in text:
                    idx_list, idx_mask, adj_mask, adj_matx = self.tokenizer.encode(t)
                    # Pad or truncate to max_length
                    if len(idx_list) > max_length:
                        idx_list = idx_list[:max_length]
                        idx_mask = idx_mask[:max_length]
                    else:
                        pad_len = max_length - len(idx_list)
                        idx_list.extend([self.tokenizer.pad_id] * pad_len)
                        idx_mask.extend([1] * pad_len)  # 1 for padding mask
                    
                    all_ids.append(idx_list)
                    all_masks.append([1 - m for m in idx_mask])  # Convert to attention mask (1 for valid tokens)
                
                text_tokens_ids = torch.tensor(all_ids)
                attention_mask = torch.tensor(all_masks)
            else:
                idx_list, idx_mask, adj_mask, adj_matx = self.tokenizer.encode(text)
                # Pad or truncate to max_length
                if len(idx_list) > max_length:
                    idx_list = idx_list[:max_length]
                    idx_mask = idx_mask[:max_length]
                else:
                    pad_len = max_length - len(idx_list)
                    idx_list.extend([self.tokenizer.pad_id] * pad_len)
                    idx_mask.extend([1] * pad_len)
                
                text_tokens_ids = torch.tensor(idx_list)
                attention_mask = torch.tensor([1 - m for m in idx_mask])  # Convert to attention mask
        else:
            # For other tokenizers (scibert, pubmed, molT5)
            text_input = self.tokenizer(
                text, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
            text_tokens_ids = text_input['input_ids'].squeeze()
            attention_mask = text_input['attention_mask'].squeeze()
        
        return text_tokens_ids, attention_mask

    def forward(self, input_ids, attention_mask=None, if_eval=True):
        if if_eval:
            self.model.eval()
            with torch.no_grad():
                if self.model_name == 'scibert':
                    typ = torch.zeros(input_ids.shape).long().to(input_ids.device)
                    output = self.model(input_ids, token_type_ids=typ, attention_mask=attention_mask)[0]
                elif self.model_name == 'pubmed':
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
                elif self.model_name == 'molT5':
                    output = self.model.encoder(input_ids).last_hidden_state
                elif self.model_name == 'chembert':
                    output = self.model(input_ids)
                else:
                    raise ValueError(f"Unsupported model name: {self.model_name}")
        else:
            if self.model_name == 'scibert':
                typ = torch.zeros(input_ids.shape).long().to(input_ids.device)
                output = self.model(input_ids, token_type_ids=typ, attention_mask=attention_mask)[0]
            elif self.model_name == 'pubmed':
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            elif self.model_name == 'molT5':
                output = self.model.encoder(input_ids).last_hidden_state
            elif self.model_name == 'chembert':
                output = self.model(input_ids)

            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")
        return self.dropout(output)
