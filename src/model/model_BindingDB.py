import os
import sys
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_max_pool as gmp, global_mean_pool as gep


porj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(porj_path)
from src.model.prompt_fusion import TwoStagePromptFusion

from src.model.ban import KBANLayer
from src.model.light import  LiGhTPredictor as LiGhT

warnings.filterwarnings("ignore")
phar_list = ['Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZincBinder', 'Aromatic', 'Hydrophobe',
             'LumpedHydrophobe']
phar_num_list = [1, 1, 1, 4, 6, 5, 2, 7]


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

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
    

def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, d_hidden_feats=None):
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
    predictor.apply(lambda module: init_params(module))
    return predictor

def init_model(args, device, config, vocab, train_dataset,text_model=None):
    base_config = config['base']
    text_encoder_config = config['text_encoder'][args.text_model_name]
    smile_encoder_config = config['smiles_encoder'][args.smiles_model_name]

    model = LiGhT(
        d_node_feats=base_config['d_node_feats'],
        d_edge_feats=base_config['d_edge_feats'],
        d_g_feats=base_config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=base_config['d_hpath_ratio'],
        n_mol_layers=base_config['n_mol_layers'],
        path_length=base_config['path_length'],
        n_heads=base_config['n_heads'],
        n_ffn_dense_layers=base_config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=args.dropout,
        feat_drop=args.dropout,
        n_node_types=vocab.vocab_size
    ).to(device)
        
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}',map_location='cpu').items()})

    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor

    # d_hidden_feats = 256
    # model.node_prompt_proj = MLP(base_config['d_g_feats'], base_config['d_g_feats'], args.projection_layers, nn.GELU(), d_hidden_feats=d_hidden_feats, dropout=args.dropout).to(device)
    # model.graph_other_prompt_proj = MLP(base_config['d_g_feats'], text_encoder_config['out_dim'], args.projection_layers, nn.GELU(), d_hidden_feats=d_hidden_feats, dropout=args.dropout).to(device)
    model.text_question_prompt_proj = MLP(text_encoder_config['out_dim'], text_encoder_config['out_dim'], args.projection_layers, nn.GELU(), d_hidden_feats=text_encoder_config['out_dim'], dropout=args.dropout).to(device)  
    model.text_disp_prompt_proj = MLP(text_encoder_config['out_dim'], text_encoder_config['out_dim'], args.projection_layers, nn.GELU(), d_hidden_feats=text_encoder_config['out_dim'], dropout=args.dropout).to(device)
    model.smiles_embed_proj = MLP(smile_encoder_config['out_dim'], text_encoder_config['out_dim'], args.projection_layers, nn.GELU(), d_hidden_feats=text_encoder_config['out_dim'], dropout=args.dropout).to(device)
    model.bcn_list = KBANLayer(
        v_dim=base_config['d_g_feats'], 
        q_dim=text_encoder_config['out_dim'], 
        k_dim=text_encoder_config['out_dim'],
        h_dim=config['BAN']['out_dim'], 
        dropout=args.dropout, 
        act=nn.GELU(), 
        k=3, 
        h_out=2, 
    ).to(device)
    
    model.prompt_fusion = TwoStagePromptFusion(
        dim=config['BAN']['out_dim'],  # 使用BAN的
        dim_out=config['head']['out_dim'],
        phar_num_list=phar_num_list,   # 8个药效团组
        dropout=args.dropout,
        projection_layers=args.projection_layers,
        d_hidden_feats=config['head']['out_dim']
    ).to(device)

    model.prompt_linear_model = get_predictor(d_input_feats=config['BAN']['out_dim'], n_tasks=1, n_layers=args.projection_layers,
                                                predictor_drop=args.dropout,d_hidden_feats=256)
    # model.predictor = get_predictor(
    #     d_input_feats=config['head']['out_dim'] + base_config['d_g_feats'],  # 8 * D 维的输入
    #     n_tasks=train_dataset.n_tasks,
    #     n_layers=args.n_layers,
    #     predictor_drop=args.dropout,
    #     device=device,
    #     d_hidden_feats=256
    # )
    if args.train_text_model:
        model.text_model = text_model.to(device)
    return model

class pharmagent_CPI(torch.nn.Module):
    def __init__(self, args, device, config, d_fps, d_mds, vocab, emb_size, max_length, n_output=1,
                 hidden_size=128, num_features_mol=78, train_dataset=None, text_model=None, dropout=None):
        super(pharmagent_CPI, self).__init__()

        # print('CPI_regression model Loaded..')
        self.device = device
        self.skip = 1
        self.n_output = n_output
        self.max_length = max_length
        self.args = args

        # proteins network
        self.prot_rnn = nn.LSTM(emb_size, hidden_size, 1)
        self.relu = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)

        # compounds network
        if args.use_encoder == 'LiGhT':
            model = LiGhT(
                d_node_feats=config['d_node_feats'],
                d_edge_feats=config['d_edge_feats'],
                d_g_feats=config['d_g_feats'],
                d_fp_feats=d_fps,
                d_md_feats=d_mds,
                d_hpath_ratio=config['d_hpath_ratio'],
                n_mol_layers=config['n_mol_layers'],
                path_length=config['path_length'],
                n_heads=config['n_heads'],
                n_ffn_dense_layers=config['n_ffn_dense_layers'],
                input_drop=0,
                attn_drop=config['dropout'],
                feat_drop=config['dropout'],
                n_node_types=vocab.vocab_size,
            )

            self.embedding_model = model

            self.prot_comp_mix = nn.Sequential(nn.Linear((config['d_g_feats'] * 3 + 1) * (hidden_size + 1), 1024),
                                               nn.LeakyReLU(),
                                               nn.Dropout(dropout))
            self.fc = nn.Sequential(nn.Linear(1024 + (config['d_g_feats'] * 3 + 1) + (hidden_size + 1), 512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(512, self.n_output))
        elif args.use_encoder == 'pharmaQA':
            model = init_model(args, device, config, vocab, train_dataset, text_model=None)
            self.embedding_model = model
            
            self.prot_comp_mix = nn.Sequential(
                nn.Linear((config['head']['out_dim'] * 4 + 1) * (hidden_size + 1), 1024),
                nn.LeakyReLU(),
                nn.Dropout(dropout))
            self.fc = nn.Sequential(nn.Linear(1024 + (config['head']['out_dim'] * 4 + 1) + (hidden_size + 1), 512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(512, self.n_output))
                
    def forward(self, ecfp, md, g, proteins, protein_lens, text_dict, smiles_embedding, smiles_mask):

        '''use KPGT encoder'''
        # Forward pass through prompt model
        molecules_repr, pred_phar_num, atten =  self.embedding_model.forward_pharmaQA_BindingDB(
           g, ecfp, md, text_dict, smiles_embedding, smiles_mask
        )

        # molecule_repr = output_repr
        
        # prompt_27_feat, att_vq_list, att_vk_list, att_qk_list = self.embedding_model.forward_tune(g, ecfp, md,
        #                                                                                             text_dict,
        #                                                                                             smiles_embed=smiles_embedding,
        #                                                                                             smiles_mask=smiles_mask,
        #                                                                                             softmax=True)
        # # Create pharma-specific features for different types (8 types assumed)
        # # non-ensemble Learning
        # start = 0
        # molecules_phar_prompt = []
        # for pharma_index in range(8):
        #     molecules_phar_prompt.append(self.embedding_model.pharma_projection_model_list[pharma_index](
        #         prompt_27_feat[:, start:start + phar_num_list[pharma_index], :].reshape(batch_size, -1)))

        # molecules_prompt = torch.stack(molecules_phar_prompt, dim=1)

        # # mlp pooling
        # molecules_prompt = self.embedding_model.prompt_projection_model(molecules_prompt.reshape(batch_size, -1))
        # molecules_repr = torch.cat((molecules_prompt, molecule_repr), dim=-1)

        # # Predict pharmaceutical number
        # pred_phar_num = self.embedding_model.prompt_linear_model(prompt_27_feat).to(torch.float32).squeeze(-1)

        # atten = torch.stack(att_vq_list, dim=2).sum(dim=-1) 

        # protein network
        pro_seq_lengths, pro_idx_sort = torch.sort(protein_lens.view(-1), descending=True)[::-1][1], torch.argsort(
            -protein_lens.view(-1))
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        proteins = proteins.index_select(0, pro_idx_sort)
        xt = nn.utils.rnn.pack_padded_sequence(proteins, pro_seq_lengths.cpu(), batch_first=True)
        xt, _ = self.prot_rnn(xt)
        xt = nn.utils.rnn.pad_packed_sequence(xt, batch_first=True, total_length=max(1500, self.max_length))[0]
        xt = xt.index_select(0, pro_idx_unsort)

        # fusion module
        xt = xt.mean(1)
        atten_protein_ligand = None
        # kronecker product
        prot_out = torch.cat((xt, torch.FloatTensor(xt.shape[0], 1).fill_(1).to(self.device)), 1)
        comp_out = torch.cat(
            (molecules_repr, torch.FloatTensor(molecules_repr.shape[0], 1).fill_(1).to(self.device)),
            1)
        output = torch.bmm(prot_out.unsqueeze(2), comp_out.unsqueeze(1)).flatten(start_dim=1)
        output = self.dropout(output)
        embeddings = self.prot_comp_mix(output)

        if self.skip:
            embeddings = torch.cat((embeddings, prot_out, comp_out), 1)

        output = self.fc(embeddings)

        return output, pred_phar_num, atten, atten_protein_ligand, embeddings

    def get_text_repr(self, text_tokens_ids, text_masks):
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_output
        return text_repr

    def get_protein_repr(self, data_pro, data_pro_len):
        pro_seq_lengths, pro_idx_sort = torch.sort(data_pro_len.view(-1), descending=True)[::-1][1], torch.argsort(
            -data_pro_len.view(-1))
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        data_pro = data_pro.index_select(0, pro_idx_sort)
        xt = nn.utils.rnn.pack_padded_sequence(data_pro, pro_seq_lengths.cpu(), batch_first=True)
        xt, _ = self.prot_rnn(xt)
        xt = nn.utils.rnn.pad_packed_sequence(xt, batch_first=True, total_length=max(1500, self.max_length))[0]
        xt = xt.index_select(0, pro_idx_unsort)
        xt = xt.mean(1)

        return xt

    def get_embeddings(self, ecfp, md, g, data_mol, text, text_mask, data_pro, data_pro_len):
        if self.args.use_encoder == 'pharmaVQA_prompt':
            '''use KPGT encoder'''
            g_1 = deepcopy(g)
            molecules_phar_prompt, atten = self.embedding_model.forward_tune(g, ecfp, md, text, text_mask)
            molecules_prompt = self.embedding_model.prompt_projection_model(
                molecules_phar_prompt.reshape(molecules_phar_prompt.shape[0], -1))

            # graph input
            molecule_repr = self.embedding_model.get_graph_feat(g_1, ecfp, md)
            molecules_repr = torch.cat((molecules_prompt, molecule_repr), dim=-1)
            # pharVQA count loss
            pred_phar_num = self.embedding_model.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(
                -1)
        elif self.args.use_encoder == 'pharmaVQA':
            '''use KPGT encoder'''
            # graph input
            molecules_repr = self.embedding_model.get_graph_feat(g, ecfp, md)
            # pharVQA count loss
            pred_phar_num = None
            atten = None
        elif self.args.use_encoder == 'SAGE':
            '''use CPI encoder'''
            # extract molecules representation
            mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch

            graph_rep = self.g1(mol_x, mol_edge_index)
            graph_rep = self.dropout(graph_rep)

            unique_labels, counts = torch.unique(mol_batch, return_counts=True)
            molecule_node_repr = pad_sequence(torch.split(graph_rep, counts.tolist()),
                                              batch_first=False, padding_value=-999)

            vv2 = molecule_node_repr.new_ones(
                (max(g.batch_num_nodes().tolist()), len(counts), molecule_node_repr.shape[2])) * -999
            vv2[:molecule_node_repr.shape[0], :, :] = molecule_node_repr
            molecule_node_repr = vv2.transpose(0, 1)
            molecule_node_mask = (molecule_node_repr[:, :, 0] != -999).float()

            # text
            question_nums = text.size()[1]
            text_reprs = [self.text_prompt_proj(self.get_text_repr(text[:, idx, :], text_mask[:, idx, :]))
                          for idx in range(question_nums)]
            # BAN network
            logits_list, atten = [], []
            for question_idx in range(question_nums):
                logits_vad, att = self.bcn(molecule_node_repr, text_reprs[question_idx], molecule_node_mask,
                                           text_mask[:, question_idx, :], softmax=True)
                logits_list.append(logits_vad)
                atten.append(att[:, -1, :, :])

            # fusion
            molecules_phar_prompt = torch.stack(logits_list).transpose(0, 1)
            molecules_prompt = self.prompt_projection_model(
                molecules_phar_prompt.reshape(molecules_phar_prompt.shape[0], -1))

            graph_rep = self.g2(mol_x, mol_edge_index)
            molecules_rep = gmp(graph_rep, mol_batch)  # global max pooling
            molecules_rep = self.dropout(molecules_rep)

            # cat
            molecules_repr = torch.cat((molecules_prompt, molecules_rep), dim=-1)
            pred_phar_num = self.prompt_linear_model(molecules_phar_prompt).to(torch.float64).squeeze(-1)

        # protein network
        pro_seq_lengths, pro_idx_sort = torch.sort(data_pro_len.view(-1), descending=True)[::-1][1], torch.argsort(
            -data_pro_len.view(-1))
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        data_pro = data_pro.index_select(0, pro_idx_sort)
        xt = nn.utils.rnn.pack_padded_sequence(data_pro, pro_seq_lengths.cpu(), batch_first=True)
        xt, _ = self.prot_rnn(xt)
        xt = nn.utils.rnn.pad_packed_sequence(xt, batch_first=True, total_length=max(1500, self.max_length))[0]
        xt = xt.index_select(0, pro_idx_unsort)

        # fusion module
        if self.args.use_attention:
            molecules_reprs = [torch.cat((phar_prompt, molecule_repr), dim=-1) for phar_prompt in
                               molecules_phar_prompt.transpose(0, 1)]
            molecules_reprs_ = torch.stack(molecules_reprs).transpose(0, 1)

            protein_mask = torch.ones((data_pro_len.shape[0], xt.shape[1]), device=self.args.device)
            molecules_mask = torch.ones((data_pro_len.shape[0], molecules_reprs_.shape[1]), device=self.args.device)
            for i in range(data_pro_len.shape[0]):
                protein_mask[i, pro_seq_lengths[i]:] = 0  # 将超过有效长度的部分设为 0

            # output, atten_protein_ligand = self.atten_encoder(xt, molecules_reprs_, protein_mask, molecules_mask)

            output, atten_protein_ligand = self.bcn(molecules_reprs_, xt, molecules_mask, protein_mask, softmax=True)
            output = self.dropout(output)
            output = self.prot_comp_mix(output)

            #
            if self.skip:
                output = torch.cat((output, xt.mean(1), molecule_repr), 1)

        else:
            xt = xt.mean(1)
            atten_protein_ligand = None
            # kronecker product
            prot_out = torch.cat((xt, torch.FloatTensor(xt.shape[0], 1).fill_(1).to(self.device)), 1)
            comp_out = torch.cat(
                (molecules_repr, torch.FloatTensor(molecules_repr.shape[0], 1).fill_(1).to(self.device)),
                1)
            output = torch.bmm(prot_out.unsqueeze(2), comp_out.unsqueeze(1)).flatten(start_dim=1)
            output = self.dropout(output)
            output = self.prot_comp_mix(output)

            if self.skip:
                output = torch.cat((output, prot_out, comp_out), 1)
        # output = self.fc(output)

        return output

