import json
import os
import sys
from time import strftime

import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from src.model.ban import KBANLayer
from src.model.ban import MLP
from src.utils import set_random_seed
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
import numpy as np
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import PharmaQADataset
from src.data.collator import Collator_pharmagent 
from src.model.light import LiGhTPredictor as LiGhT,TextEncoder
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.finetune_trainer import Trainer_pharmaQA
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict
from src.model.prompt_fusion import TwoStagePromptFusion

import warnings
import torch.multiprocessing

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
phar_num_list = [1, 1, 1, 4, 6, 5, 2, 7]
SPLIT_METHOD_ALIASES = {
    'scaffold_345': 'scaffold',
    'kpgt_vs': 'scaffold',
}
VALID_SPLIT_METHODS = ['random', 'kpgt', 'scaffold', 'ace', 'commn', 'MoleculeNet', 'vs']

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def str2bool(inp):
    inp = inp.lower()
    if inp in ['y', 'yes', 'true', 't']:
        return True
    else:
        return False
    
def attention_alignment_loss(attention_source, attention_target):
    cos_sim = F.cosine_similarity(attention_source, attention_target, dim=-1)
    return (1.0 - cos_sim).mean()

def save_results(results_list, args):
    df = pd.DataFrame(results_list)
    if args.ablation_mode_flag:
        file_path = f'{args.save_path}/{args.dataset}_questionNum{args.num_questions}_{args.split}_base_encoder_{args.base_model_encoder}_ablation_mode_{args.ablation_mode}_seed_{args.seed}_finetune.csv'
    else:
        file_path = f'{args.save_path}/{args.dataset}_questionNum{args.num_questions}_{args.split}_base_encoder_{args.base_model_encoder}_seed_{args.seed}_finetune.csv'
    df.to_csv(file_path, mode='a', index=False, header=False)
    print(f"Results saved to {file_path}")

def create_save_path(args, seed):
    """Creates and returns the save path for storing model checkpoints."""
    if args.ablation_mode_flag:
        save_path = f'{base_path}/save/{args.dataset}/question_num{args.num_questions}/{args.split}/seed_{seed}/text_model_{args.text_model_name}/ablation_mode_{args.ablation_mode}/base_encoder_{args.base_model_encoder}/alpha_{args.alpha}_beta_{args.beta}/'
    else:
        save_path = f'{base_path}/save/{args.dataset}/question_num{args.num_questions}/{args.split}/seed_{seed}/text_model_{args.text_model_name}/base_encoder_{args.base_model_encoder}/alpha_{args.alpha}_beta_{args.beta}/'
    if args.split_method == 'ace':
        save_path = f'{base_path}/save/{args.dataset}/question_num{args.num_questions}/{args.split}/seed_{seed}/text_model_{args.text_model_name}/base_encoder_{args.base_model_encoder}/alpha_{args.alpha}_beta_{args.beta}/ace_clip_test_{args.ace_clip_test}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(f"{save_path}/tensorboard"):
        os.makedirs(f"{save_path}/tensorboard")
    return save_path

def collect_and_save_all_results(results_dict, args):
    project_name = args.split_method
    
    if args.ablation_mode_flag:
        output_file = f'{base_path}/save/all_results_{project_name}_questionNum{args.num_questions}_base_encoder_{args.base_model_encoder}_ablation_mode_{args.ablation_mode}_seed_{args.seed}_finetune.csv'
    else:
        output_file = f'{base_path}/save/all_results_{project_name}_questionNum{args.num_questions}_base_encoder_{args.base_model_encoder}_seed_{args.seed}_finetune.csv'
    
    all_results = []
    
    metrics = args.metric.split(',')
    
    for dataset_name, results in results_dict.items():
        for result in results:
            result_dict = {
                'dataset': dataset_name,
                'split': result[0],
                'run': result[1],
            }
            
            if isinstance(result[2], (tuple, list)):
                for i, metric_value in enumerate(result[2]):
                    if i < len(metrics):
                        result_dict[metrics[i]] = metric_value
            else:
                result_dict[metrics[0]] = result[2]
            
            all_results.append(result_dict)
    
    df = pd.DataFrame(all_results)
    
    df.to_csv(output_file, index=False)
    print(f"All results saved to: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--mode", type=str, default='finetune', choices=['finetune', 'evaluate'],
                       help='Mode to run: finetune for training, evaluate for inference only')

    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_path", type=str, default=f'{base_path}/pretrained/base/base.pth')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--data_path", type=str, default=f'{base_path}/datasets/ligand')
    parser.add_argument(
        "--split_method",
        type=str,
        default='scaffold',
        help="Split strategy. Supported values: random, kpgt, scaffold, ace, commn, MoleculeNet, vs. Legacy aliases: scaffold_345, kpgt_vs",
    )
    parser.add_argument("--dataset_idx_start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.00003, help='model learning rate for training (default: 0.00003)')

    parser.add_argument("--num_workers", type=int, default=0, 
                       help='Number of workers for data loading. Set to 0 to avoid multiprocessing issues.')

    # PharmAgent
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--projection_layers", type=int, default=2)
    parser.add_argument("--num_questions", type=int, default=len(phar_num_list))

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--ablation_mode_flag", type=str2bool, default=False)
    parser.add_argument("--text_model_name", type=str, default='pubmed')
    parser.add_argument("--smiles_model_name", type=str, default='chembert')
    parser.add_argument("--train_text_model", type=str2bool, default=False)
    parser.add_argument("--ablation_mode", type=str, default='no_prompt', choices=['no_prompt', 'noise_prompt'])
    parser.add_argument("--phar_load_method", type=str, default='lmdb', choices=['pkl', 'lmdb'])
    parser.add_argument("--smiles_load_method", type=str, default='lmdb', choices=['pkl', 'lmdb'])
    parser.add_argument("--train_kpgt", type=str2bool, default=False)


    parser.add_argument("--base_model_encoder", type=str, default='LiGhT')
    parser.add_argument("--prompt_graph_feature_extractor", type=str, default='LiGhT')
    parser.add_argument("--ace_clip_test", type=str2bool, default=False)
    parser.add_argument("--use_norm_reg", type=str2bool, default=False)

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)    
    parser.add_argument("--use_phar_loss", type=str2bool, default=True)
    parser.add_argument("--use_align_loss", type=str2bool, default=True)
    parser.add_argument("--noise_question", type=str, default='To be, or not to be, that is the question.')

    parser.add_argument("--use_amp", type=str2bool, default=False,
                       help='Whether to use Automatic Mixed Precision training')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help='Number of updates steps to accumulate before performing a backward/update pass')

    parser.add_argument("--lr_warmup_epochs", type=int, default=10,
                       help='Number of epochs for learning rate warmup')
    parser.add_argument("--min_lr", type=float, default=1e-9,
                       help='Minimum learning rate')

    args = parser.parse_args()
    args.split_method = SPLIT_METHOD_ALIASES.get(args.split_method, args.split_method)
    if args.split_method not in VALID_SPLIT_METHODS:
        parser.error(f"invalid --split_method '{args.split_method}'")
    return args


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
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)


def get_question_embeddings(args):
    text_path = os.path.join(args.data_path, '../text', 'phar_question_howmany_gpt4o_27.json')

    with open(text_path, 'r', encoding='utf-8') as fp:
        text_list = json.load(fp)
    # Initialize lists for selected questions and pharmacophore names
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
                    smart = item['SMARTS']
                    # combined_text = f"Question: {question}, Description: {description}"
                    question_text = f"Question: {question}"
                    discription_text = f"Description: {description}"
                    # discription_text = f"SMART: {smart}, Description: {description}"
                    select_question.append(question_text)
                    select_discription.append(discription_text)
                    phar_question_name.append(item['type'])
    else:
        # Iterate through the text list and extract questions and descriptions
        for category, items in text_list.items():
            for item in items:
                question = item['question']
                description = item['description']
                smart = item['SMARTS']
                # combined_text = f"Question: {question}, Description: {description}"
                question_text = f"Question: {question}"
                discription_text = f"Description: {description}"
                # discription_text = f"SMART: {smart}, Description: {description}"
                select_question.append(question_text)
                select_discription.append(discription_text)
                phar_question_name.append(item['type'])

    # Tokenize the selected questions and descriptions

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

def init_dataset(args, g, collator, phar_question_name):
    dataset_params = {
        'root_path': args.data_path,
        'dataset': args.dataset,
        'dataset_type': args.dataset_type,
        'train_kpgt': args.train_kpgt,
        'split_name': f'{args.split}',
        'phar_question_name': phar_question_name,
        'use_norm_reg': args.use_norm_reg,
        'smi_encoder_name': args.smiles_model_name,
        'phar_load_method': args.phar_load_method,
        'smiles_load_method': args.smiles_load_method,
        'split_method': args.split_method,
        'base_model_encoder': args.base_model_encoder,
        'ace_clip_test': args.ace_clip_test,
        'prompt_graph_feature_extractor': args.prompt_graph_feature_extractor,
        'seed': args.seed
    }

    train_dataset = PharmaQADataset(split='train', **dataset_params)
    val_dataset = PharmaQADataset(split='val', **dataset_params)
    test_dataset = PharmaQADataset(split='test', **dataset_params)

    print(f"\nDataset sizes:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples\n")

    # 设置多进程参数，避免序列化问题
    multiprocessing_kwargs = {}
    if args.num_workers > 0:
        # 设置多进程共享策略
        torch.multiprocessing.set_sharing_strategy('file_system')
        multiprocessing_kwargs = {
            'num_workers': args.num_workers,
            'worker_init_fn': seed_worker,
            'generator': g,
        }
    else:
        # 单进程模式
        multiprocessing_kwargs = {
            'num_workers': 0,
            'generator': g,
        }

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        **multiprocessing_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        **multiprocessing_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        **multiprocessing_kwargs
    )

    return train_loader, val_loader, test_loader, train_dataset

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
    h_out=2,    
    dropout=args.dropout, 
    act=nn.GELU(), 
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
    
    # model.base_feat_fusion = MLP(3 * base_config['d_g_feats'], base_config['d_g_feats'], args.projection_layers, nn.GELU(), d_hidden_feats=256, dropout=args.dropout)
    model.predictor = get_predictor(
        d_input_feats=config['head']['out_dim'] + 3*base_config['d_g_feats'],
        n_tasks=train_dataset.n_tasks,
        n_layers=args.n_layers,
        predictor_drop=args.dropout,
        device=device,
        d_hidden_feats=256
    )
    if args.train_text_model:
        model.text_model = text_model.to(device)
    return model

def init_optimizer_and_scheduler(args, model, train_dataset):
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.n_epochs * len(train_dataset) // args.batch_size
    num_warmup_steps = args.lr_warmup_epochs * len(train_dataset) // args.batch_size

    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=num_warmup_steps,
        tot_updates=num_training_steps,
        lr=args.lr,
        end_lr=args.min_lr,
        power=1
    )

    return optimizer, lr_scheduler

def finetune(args, seed):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Finetuning on dataset {args.dataset}, split {args.split}, seed {seed}, lr {args.lr}")
    print(f"Using AMP: {args.use_amp}")

    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    save_path = create_save_path(args, seed)
    args.save_path = save_path
    collator = Collator_pharmagent(args.max_length)

    # get text question embeddings
    text_dict, text_model = get_question_embeddings(args)

    train_loader, val_loader, test_loader, train_dataset = init_dataset(args, g, collator, text_dict['phar_question_name'])
    # Model Initialization
    model = init_model(args, device, config_dict, vocab, train_dataset,text_model)
    # Finetuning Setting
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))
    print('text_model have {}M paramerters in total'.format(sum(x.numel() for x in text_model.parameters()) / 1e6))
    if not args.train_text_model:
        text_model.eval()
        print('text_model is trainable: {}'.format(False))
    else:
        print('text_model is trainable: {}'.format(True))
    optimizer, lr_scheduler = init_optimizer_and_scheduler(args, model, train_dataset)

    if args.dataset_type == 'classification':
        loss_fn = BCEWithLogitsLoss(reduction='none')
    else:
        loss_fn = MSELoss(reduction='none')

    phar_loss_fn = MSELoss(reduction='none')
    align_loss_fn = attention_alignment_loss
    if args.dataset_type == 'classification':
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks)
    else:
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks, mean=train_dataset.mean.numpy() if train_dataset.mean is not None else None,
                              std=train_dataset.std.numpy() if train_dataset.std is not None else None)
    phar_evaluator = Evaluator(args.dataset, 'rmse', 27)

    result_tracker = Result_Tracker(args.metric)

    starttime = strftime("%Y-%m-%d_%H-%M-%S")
    summary_writer = SummaryWriter(
        f"{args.save_path}/tensorboard/st{starttime}",
        comment=starttime
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    trainer = Trainer_pharmaQA(args, optimizer, lr_scheduler, loss_fn, phar_loss_fn, align_loss_fn, evaluator,
                           phar_evaluator, result_tracker,
                           summary_writer, device_id=device,
                           label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None,
                           label_std=train_dataset.std.to(device) if train_dataset.std is not None else None,
                           scaler=scaler)
    
    # 添加：在训练完成后提取分子表示向量
    best_train, best_val, best_test = trainer.fit(model, train_loader, val_loader, test_loader, text_dict)
    
    # 新增：提取分子表示向量用于可视化分析
    # print("Extracting molecular representations for visualization...")
    # extract_molecular_representations(model, test_loader, text_dict, device, save_path, train_dataset)
    
    # Print results
    if ',' in args.metric:
        for index, metric in enumerate(args.metric.split(',')):
            print(
                f"Train_{metric}: Val_{metric}: {best_val[index]:.3f}, Test_{metric}: {best_test[index]:.3f}")

    else:
        print(f"Train: {best_train:.3f}, Val: {best_val:.3f}, Test: {best_test:.3f}")

    return best_test


def evaluation(args, seed=None):
    """评估函数，用于在预训练模型上进行推理"""
    if seed is None:
        seed = args.seed
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on dataset {args.dataset}, split {args.split}, seed {seed}")
    
    # 初始化词汇表和生成器
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(seed)
    
    # 创建保存路径
    save_path = create_save_path(args, seed)
    args.save_path = save_path
    # Keep evaluation and finetuning on the same collator to avoid
    # diverging batch formats between the two code paths.
    collator = Collator_pharmagent(args.max_length)

    # 获取文本问题嵌入
    text_dict, text_model = get_question_embeddings(args)

    # 初始化数据集
    train_loader, val_loader, test_loader, train_dataset = init_dataset(args, g, collator, text_dict['phar_question_name'])
    
    # 初始化模型
    model = init_model(args, device, config_dict, vocab, train_dataset, text_model)
    
    # 设置为评估模式
    model.eval()
    print("Model have {}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))
    print('Text model have {}M parameters in total'.format(sum(x.numel() for x in text_model.parameters()) / 1e6))
    if not args.train_text_model:
        text_model.eval()
        print('Text model is trainable: {}'.format(False))
    else:
        print('Text model is trainable: {}'.format(True))
    
    # 初始化损失函数和评估器 - 与finetune保持一致
    if args.dataset_type == 'classification':
        loss_fn = BCEWithLogitsLoss(reduction='none')
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks)
    else:
        loss_fn = MSELoss(reduction='none')
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks, 
                            mean=train_dataset.mean.numpy() if train_dataset.mean is not None else None,
                            std=train_dataset.std.numpy() if train_dataset.std is not None else None)
    
    phar_loss_fn = MSELoss(reduction='none')
    align_loss_fn = attention_alignment_loss
    phar_evaluator = Evaluator(args.dataset, 'rmse', 27)
    
    # 初始化结果跟踪器
    result_tracker = Result_Tracker(args.metric)
    
    # 创建tensorboard writer
    starttime = strftime("%Y-%m-%d_%H-%M-%S")
    summary_writer = SummaryWriter(
        f"{args.save_path}/tensorboard/st{starttime}",
        comment=starttime
    )
    
    # 创建scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # 创建trainer - 与finetune保持一致
    trainer = Trainer_pharmaQA(args, None, None, loss_fn, phar_loss_fn, align_loss_fn, evaluator,
                           phar_evaluator, result_tracker,
                           summary_writer, device_id=device,
                           label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None,
                           label_std=train_dataset.std.to(device) if train_dataset.std is not None else None,
                           scaler=scaler)
    
    # 使用trainer进行评估
    print("Starting evaluation on test set...")
    test_results, predictions_all, labels_all = trainer.eval(model, test_loader, 0, text_dict)
    
    print(f"Evaluation completed!")
    
    # 保存预测结果
    if (train_dataset.mean is not None) and (train_dataset.std is not None):
        predictions_all = predictions_all * train_dataset.std.detach().cpu() + train_dataset.mean.detach().cpu()
    
    np.savetxt(f'{args.save_path}/evaluation_results.txt',
               np.concatenate([predictions_all, labels_all], axis=1),
               fmt='%.4f',
               header='Predictions,Labels')
    
    # Print results - 与finetune保持一致
    if ',' in args.metric:
        for index, metric in enumerate(args.metric.split(',')):
            print(f"Test_{metric}: {test_results[index]:.3f}")
    else:
        print(f"Test: {test_results:.3f}")
    
    return test_results


def run_experiment(args, mode='finetune'):
    results_list = []
    for runs in range(args.num_runs):
        seed = args.seed + runs
        set_random_seed(seed)

        if mode == 'evaluate':
            results = evaluation(args, seed)
        else:
            results = finetune(args, seed)

        results_list.append([args.split, runs, results])
        print(f'Dataset {args.dataset} on {args.split} at seed {seed} results is: {results}')

    return results_list



def setup_dataset_config(dataset_name):
    config = {
        'dataset_type': 'classification',
        'metric': 'rocauc'
    }
    
    if dataset_name in ['lipo', 'esol', 'freesolv']:
        config['dataset_type'] = 'regression'
        config['metric'] = 'rmse,r2'
    elif dataset_name in ['EGFR','JAK1']:
        config['dataset_type'] = 'regression'
        config['metric'] = 'rmse,r2'
    elif any(dataset_name.startswith(prefix) for prefix in ['HPK1_IC50', 'FGFR1_IC50', 'VIM1_IC50']):
        config['dataset_type'] = 'regression'
        config['metric'] = 'spear,pear'
    return config

def handle_split_method(args, data, all_results_dict,mode='finetune'):
    if args.split_method == 'random':
        for split_idx in range(args.split_start, args.split_runs):
            args.split = f'scaffold-random{split_idx}'
            results_list = run_experiment(args, mode=mode)
            save_results(results_list, args)
            
    elif args.split_method == 'kpgt':
        for split_idx in range(3):
            args.split = f'scaffold-{split_idx}'
            results_list = run_experiment(args, mode=mode)
            save_results(results_list, args)
            
    elif args.split_method == 'scaffold':
        for split_idx in [3,4,5]:
            args.split = f'scaffold-{split_idx}'
            results_list = run_experiment(args, mode=mode)
            save_results(results_list, args)
            
    elif args.split_method == 'commn':
        args.split = 'scaffold-commn'
        results_list = run_experiment(args, mode=mode)
        save_results(results_list, args)
        
    elif args.split_method == 'MoleculeNet':
        if args.dataset not in []:
            args.num_runs = 10
            args.split = 'scaffold-MoleculeNet'
            results_list = run_experiment(args, mode=mode)
            save_results(results_list, args)
            
    elif args.split_method == 'ace':
        args.split = 'ace'
        args.dataset_type = 'regression'
        args.metric = 'rmse,r2'
        if data_idx >= args.dataset_idx_start:
            results_list = run_experiment(args, mode=mode)
            all_results_dict[data] = results_list
            save_results(all_results_dict[data], args)
    elif args.split_method == 'vs':
        args.split = 'scaffold-0'
        results_list = run_experiment(args, mode=mode)
        save_results(results_list, args)

    return all_results_dict

if __name__ == '__main__':
    args = parse_args()
    if args.dataset:
        data_list = [args.dataset]
    else:
        data_list = sorted(os.listdir(args.data_path))
    all_results_dict = {}
    
    print(f"Running in {args.mode} mode")
    
    for data_idx, data in enumerate(data_list):
        print(f"Processing dataset {data}")
        args.dataset = data
        config = setup_dataset_config(args.dataset)
        print(f"Dataset type: {config['dataset_type']}, Metric: {config['metric']}")
        args.dataset_type = config['dataset_type']
        args.metric = config['metric']

        all_results_dict = handle_split_method(args, data, all_results_dict,mode=args.mode)
    
    collect_and_save_all_results(all_results_dict, args)
    print(f"All results have been saved to {args.save_path}")





def generate_visualization_config(data_dir, representation_dim):
    """
    生成可视化配置文件
    """
    config = {
        'data': {
            'pharmaqa_representations': os.path.join(data_dir, 'pharmaqa_representations.npy'),
            'kpgt_representations': os.path.join(data_dir, 'kpgt_representations.npy'),  # 需要你自己提供
            'molecule_labels': os.path.join(data_dir, 'molecule_labels.npy'),
            'molecule_smiles': os.path.join(data_dir, 'molecule_smiles.csv'),
            'pharmacophore_features': os.path.join(data_dir, 'pharmacophore_features.npy'),
            'similarity_matrix': os.path.join(data_dir, 'similarity_matrix.npy'),
            'representation_dim': representation_dim,
            'n_molecules': 'auto'
        },
        'visualization': {
            'style': {
                'figure_size': [16, 12],
                'dpi': 300,
                'color_palette': 'viridis',
                'font_size': 12,
                'save_format': ['png', 'pdf']
            },
            'dimensionality_reduction': {
                'tsne': {
                    'perplexity': 30,
                    'n_iter': 1000,
                    'random_state': 42
                },
                'umap': {
                    'n_neighbors': 15,
                    'min_dist': 0.1,
                    'random_state': 42
                }
            },
            'clustering': {
                'n_clusters': 5,
                'random_state': 42
                }
        },
        'output': {
            'base_dir': os.path.join(data_dir, 'visualization_results'),
            'plots_dir': 'plots',
            'reports_dir': 'reports',
            'data_dir': 'data'
        }
    }
    
    import yaml
    config_path = os.path.join(data_dir, 'visualization_config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Visualization config saved to: {config_path}")
    print("Note: You need to provide KPGT representations for comparison!")
