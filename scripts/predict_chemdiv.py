import json
import os
import sys
import argparse
from networkx import read_graph6
import torch
import numpy as np
import pandas as pd
from time import strftime
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCEWithLogitsLoss
from tqdm import tqdm

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from dgl.data.utils import load_graphs
from src.model.ban import KBANLayer
from src.model.ban import MLP
from src.utils import set_random_seed
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import PharmaQADataset, pharmagentDataset_chemdiv
from src.data.collator import Collator_pharmagent_chemdiv
import dgl
from src.model.light import LiGhTPredictor as LiGhT, TextEncoder
from src.model.prompt_fusion import TwoStagePromptFusion
from src.model_config import config_dict

import warnings
import torch.multiprocessing

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

phar_num_list = [1, 1, 1, 4, 6, 5, 2, 7]

def str2bool(inp):
    inp = inp.lower()
    if inp in ['y', 'yes', 'true', 't']:
        return True
    else:
        return False

def init_params(module):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = torch.nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = torch.nn.ModuleList()
        predictor.append(torch.nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(torch.nn.Dropout(predictor_drop))
        predictor.append(torch.nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(torch.nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(torch.nn.Dropout(predictor_drop))
            predictor.append(torch.nn.GELU())
        predictor.append(torch.nn.Linear(d_hidden_feats, n_tasks))
        predictor = torch.nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Predict ChemDiv dataset using trained model")
    
    # 基本参数
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=128)
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default='JAK1',
                       help="Name of the model")
    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--projection_layers", type=int, default=2)
    
    # 数据路径
    parser.add_argument("--data_path", type=str, 
                       default='/workspace/aichengwei/code/own/KPGT-main/datasets/',
                       help="Path to ChemDiv dataset")

    # 模型配置
    parser.add_argument("--text_model_name", type=str, default='pubmed')
    parser.add_argument("--smiles_model_name", type=str, default='chembert')
    parser.add_argument("--train_text_model", type=str2bool, default=False)
    parser.add_argument("--ablation_mode_flag", type=str2bool, default=False)
    parser.add_argument("--ablation_mode", type=str, default='no_prompt', choices=['no_prompt', 'noise_prompt'])
    parser.add_argument("--phar_load_method", type=str, default='lmdb', choices=['pkl', 'lmdb'])
    parser.add_argument("--smiles_load_method", type=str, default='lmdb', choices=['pkl', 'lmdb'])
    parser.add_argument("--train_kpgt", type=str2bool, default=False)
    parser.add_argument("--base_model_encoder", type=str, default='LiGhT')
    parser.add_argument("--prompt_graph_feature_extractor", type=str, default='LiGhT')
    parser.add_argument("--ace_clip_test", type=str2bool, default=False)
    parser.add_argument("--use_norm_reg", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=True)
    # 其他参数
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--use_phar_loss", type=str2bool, default=True)
    parser.add_argument("--use_align_loss", type=str2bool, default=True)
    parser.add_argument("--noise_question", type=str, default='To be, or not to be, that is the question.')
    
    args = parser.parse_args()
    return args

def get_question_embeddings(args):
    text_path = os.path.join(args.data_path, './text', 'phar_question_howmany_gpt4o_27.json')

    with open(text_path, 'r', encoding='utf-8') as fp:
        text_list = json.load(fp)
    
    select_question = []
    select_discription = []
    phar_question_name = []
    
    if args.ablation_mode_flag:
        if args.ablation_mode == 'no_prompt':
            return None, None
        elif args.ablation_mode == 'noise_prompt':
            select_question = ["To be, or not to be, that is the question."]
            select_discription = ["To be, or not to be, that is the question."]
            phar_question_name = None
            for category, items in text_list.items():
                for item in items:
                    question = item['question']
                    description = item['description']
                    question_text = f"Question: {question}"
                    discription_text = f"Description: {description}"
                    select_question.append(question_text)
                    select_discription.append(discription_text)
                    phar_question_name.append(item['type'])
    else:
        for category, items in text_list.items():
            for item in items:
                question = item['question']
                description = item['description']
                question_text = f"Question: {question}"
                discription_text = f"Description: {description}"
                select_question.append(question_text)
                select_discription.append(discription_text)
                phar_question_name.append(item['type'])

    text_model = TextEncoder(model_name=args.text_model_name, load=True)
    question_texts, question_masks = text_model.tokenize(select_question, max_length=96)
    discription_texts, discription_mask = text_model.tokenize(select_discription, max_length=96)

    if question_texts.dim() == 1:
        question_texts = question_texts.unsqueeze(0)
        question_masks = question_masks.unsqueeze(0)
        discription_texts = discription_texts.unsqueeze(0)
        discription_mask = discription_mask.unsqueeze(0)

    if not args.train_text_model:
        question_text_embeddings = text_model(input_ids=question_texts, attention_mask=question_masks)
        discription_text_embeddings = text_model(input_ids=discription_texts, attention_mask=discription_mask)

        text_dict = {
            'question_text_embeddings': question_text_embeddings.to(args.device),
            'discription_text_embeddings': discription_text_embeddings.to(args.device),
            'question_masks': question_masks.to(args.device),
            'discription_mask': discription_mask.to(args.device),
            'phar_question_name': phar_question_name,
            'question_tokens': question_texts,
            'know_tokens': discription_texts,
            'question_texts': select_question,
        }
    else:
        text_dict = {
            'question_masks': question_masks.to(args.device),
            'discription_mask': discription_mask.to(args.device),
            'phar_question_name': phar_question_name,
            'question_tokens': question_texts.to(args.device),
            'know_tokens': discription_texts.to(args.device),
            'question_texts': select_question,
        }
    return text_dict, text_model

def pad_1d_list(list_of_lists, pad_value=0):
    max_len = max(len(x) for x in list_of_lists)
    padded = [list(x) + [pad_value] * (max_len - len(x)) for x in list_of_lists]
    return padded, max_len

def extract_smiles_embeddings_chembert(smiles_list, text_model):
    idx_list = []
    idx_mask_list = []
    adj_mask_list = []
    adj_matx_list = []
    for smiles in smiles_list:
        idx, idx_mask, adj_mask, adj_matx = text_model.tokenizer.encode(smiles)
        idx_list.append(idx)
        idx_mask_list.append(idx_mask)
        adj_mask_list.append(adj_mask)
        adj_matx_list.append(adj_matx)
    
    # padding
    idx_list_padded, max_len = pad_1d_list(idx_list, pad_value=0)
    idx_mask_list_padded, _ = pad_1d_list(idx_mask_list, pad_value=1)
    adj_mask_list_padded, _ = pad_1d_list(adj_mask_list, pad_value=0) 

    embeddings = text_model.model(
        input=torch.tensor(idx_list_padded, dtype=torch.long),
        imask=torch.tensor(idx_mask_list_padded, dtype=torch.bool),
        amask=torch.tensor(adj_mask_list_padded, dtype=torch.float),
        amatx=torch.tensor(adj_matx_list, dtype=torch.float),
    )

    return embeddings, torch.tensor(idx_mask_list_padded)

def init_model(args, device, config, vocab, train_dataset, text_model=None):
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
        
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor

    model.text_question_prompt_proj = MLP(text_encoder_config['out_dim'], text_encoder_config['out_dim'], args.projection_layers, torch.nn.GELU(), d_hidden_feats=text_encoder_config['out_dim'], dropout=args.dropout).to(device)  
    model.text_disp_prompt_proj = MLP(text_encoder_config['out_dim'], text_encoder_config['out_dim'], args.projection_layers, torch.nn.GELU(), d_hidden_feats=text_encoder_config['out_dim'], dropout=args.dropout).to(device)
    model.smiles_embed_proj = MLP(smile_encoder_config['out_dim'], text_encoder_config['out_dim'], args.projection_layers, torch.nn.GELU(), d_hidden_feats=text_encoder_config['out_dim'], dropout=args.dropout).to(device)

    model.bcn_list = KBANLayer(
        v_dim=base_config['d_g_feats'], 
        q_dim=text_encoder_config['out_dim'], 
        k_dim=text_encoder_config['out_dim'],
        h_dim=config['BAN']['out_dim'], 
        h_out=2,    
        dropout=args.dropout, 
        act=torch.nn.GELU(), 
        k=3).to(device)
    
    model.prompt_fusion = TwoStagePromptFusion(
        dim=config['BAN']['out_dim'],
        dim_out=config['head']['out_dim'],
        phar_num_list=phar_num_list,
        dropout=args.dropout,
        projection_layers=args.projection_layers,
        d_hidden_feats=config['head']['out_dim']
    ).to(device)
    
    model.prompt_linear_model = get_predictor(d_input_feats=config['BAN']['out_dim'], n_tasks=1, n_layers=args.n_layers,
                                            predictor_drop=args.dropout, device=device, d_hidden_feats=256)
    model.predictor = get_predictor(
        d_input_feats=config['head']['out_dim'] + base_config['d_g_feats'],
        n_tasks=1,
        n_layers=args.n_layers,
        predictor_drop=args.dropout,
        device=device,
        d_hidden_feats=256
    )
    
    if args.train_text_model:
        model.text_model = text_model.to(device)
    
    return model

def create_chemdiv_dataset(args):
    """创建ChemDiv数据集"""
    dataset_params = {
        'root_path': args.data_path,
        'dataset': 'chemdiv',
        'path_length': 5,
        'debug': args.debug,
        'seed': args.seed,
        'index': None  # 使用所有数据
    }

    dataset = pharmagentDataset_chemdiv(**dataset_params)
    return dataset

def set_mean_and_std(labels):
    """Set mean and std for normalization."""
    mean = torch.from_numpy(np.nanmean(labels.numpy(), axis=0))
    std = torch.from_numpy(np.nanstd(labels.numpy(), axis=0))
    return mean, std

def test(model, device, loader, text_model, text_dict):
    """Perform prediction using the same function as eval_pharVQA_dta_CPI_prediction.py"""
    model.eval()
    total_preds = torch.Tensor()
    smiles_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc="Predicting")):
            try:
                # Unpack the batched data from pharmagentDataset_chemdiv
                (smiles, graphs, fps, mds, phar_targets, phar_target_mx, atom_phar_target_map, labels) = data
                
                # Move tensors to the device
                fps, mds, phar_targets, phar_target_mx, atom_phar_target_map, labels = [
                    x.to(device) for x in [fps, mds, phar_targets, phar_target_mx, atom_phar_target_map, labels]
                ]
                
                # Convert graphs to device if needed
                if isinstance(graphs, list):
                    graphs = [g.to(device) if hasattr(g, 'to') else g for g in graphs]
                else:
                    graphs = graphs.to(device)
                
                # get smiles embeddings
                smiles_embeddings, smiles_mask = extract_smiles_embeddings_chembert(smiles, text_model)
                smiles_mask = 1 - smiles_mask

                smiles_embeddings = smiles_embeddings.to(device)
                smiles_mask = smiles_mask.to(device)
                
                # Forward pass through model
                predictions, pred_phar_num, atten = model.forward_pharmagent(graphs, fps, mds, text=text_dict,
                     smiles_embed=smiles_embeddings, smiles_mask=smiles_mask)

                total_preds = torch.cat((total_preds, predictions.cpu()), 0)
                smiles_list.extend(smiles)
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
    
    return smiles_list, total_preds.numpy().flatten()

def predict_chemdiv(args):
    """预测ChemDiv数据集"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # set model path
    args.model_path = f'{base_path}/save/{args.model_name}/question_num8/scaffold-0/seed_42/text_model_pubmed/base_encoder_LiGhT/alpha_0.1_beta_0.1/'
    args.model_data_graph_path = f'{base_path}/datasets/vs/{args.model_name}/{args.model_name}_5.pkl'
    args.split_path = f'{base_path}/datasets/vs/{args.model_name}/splits/scaffold-0.npy'
    args.output_dir = os.path.join(args.model_path, 'chemdiv_predictions')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 初始化词汇表
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    
    # 创建collator
    collator = Collator_pharmagent_chemdiv(args.max_length)

    # 获取文本问题嵌入
    text_dict, text_model = get_question_embeddings(args)

    # 创建ChemDiv数据集
    dataset = create_chemdiv_dataset(args)
    print(f"Loaded {len(dataset)} molecules from ChemDiv dataset")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        num_workers=args.num_workers
    )

    # 初始化模型
    model = init_model(args, device, config_dict, vocab, dataset, text_model)
    smiles_model = TextEncoder(model_name=args.smiles_model_name, load=True)
    
    # load model
    if args.model_path is not None:
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}/best_model.pth', map_location='cpu').items()})
        print(f"Model loaded from {args.model_path}/best_model.pth")
    else:
        print("Model path is not set")
    
    model.eval()
    print(f"Model loaded with {sum(x.numel() for x in model.parameters()) / 1e6:.1f}M parameters")
    print(f"Smiles model loaded with {sum(x.numel() for x in smiles_model.parameters()) / 1e6:.1f}M parameters")
    
    # 预测
    predictions = []
    smiles_list = []
    
    print("Starting prediction...")
    smiles_list, predictions = test(model, device, dataloader, smiles_model, text_dict)

    # 如果数据集有标准化参数，进行反标准化
    if args.use_norm_reg:
        try:
            _, label_dict = load_graphs(args.model_data_graph_path)
            labels = label_dict['labels']
            train_idx = np.load(args.split_path, allow_pickle=True)[0]   
            mean, std = set_mean_and_std(labels[train_idx])
            predictions = predictions * std.numpy() + mean.numpy()
            print(f"Applied normalization: mean={mean.numpy()}, std={std.numpy()}")
        except Exception as e:
            print(f"Warning: Could not apply normalization: {e}")
            print("Using raw predictions")
    
    # 保存结果
    results_df = pd.DataFrame({
        'smiles': smiles_list,
        'prediction': predictions.flatten()
    })
    
    output_path = os.path.join(args.output_dir, f'chemdiv_predictions_{strftime("%Y%m%d_%H%M%S")}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"Predicted {len(predictions)} molecules")
    
    # 显示一些统计信息
    print(f"Prediction statistics:")
    print(f"Mean: {predictions.mean():.4f}")
    print(f"Std: {predictions.std():.4f}")
    print(f"Min: {predictions.min():.4f}")
    print(f"Max: {predictions.max():.4f}")
    
    return results_df

def main():
    args = parse_args()
        
    if args.debug:
        print("Debug mode is on")
    results = predict_chemdiv(args)

    print("Prediction completed successfully!")

if __name__ == '__main__':
    main() 