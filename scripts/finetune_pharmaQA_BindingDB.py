import argparse
import json
import pickle
import sys
from collections import OrderedDict

import lmdb
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, r2_score

from datetime import datetime

import os
import warnings


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import pandas as pd
from dgl import load_graphs
from rdkit import Chem
import torch.nn as nn
from tqdm import tqdm
import scipy.sparse as sps
from src.utils import target_matrics, get_mse, smile_to_graph, get_pearson
from src.model.light import TextEncoder
from src.model.model_BindingDB import pharmaPrompt_CPI
from src.data.collator import Collator_pharVQA_CPI
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import pharmaQADataset_CPI

from src.model_config import config_dict
import _pickle as cPickle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

dropout_global = 0.2
early_stop = 5
stop_epoch = 0
best_epoch = -1
best_loss = 1000
best_mse = 100
best_auc = 0.0
best_test_mse = 100
last_epoch = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Argument")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--use_encoder', type=str, default='pharmaQA')
    parser.add_argument('--load_model_path', type=str,
                        default='')

    parser.add_argument('--train_kpgt', type=str, default='False')
    parser.add_argument('--use_atten_loss', type=str, default='False')
    parser.add_argument('--use_norm_reg', type=str, default='False')

    parser.add_argument('--align_loss', type=str, default='True')
    parser.add_argument('--phar_loss', type=str, default='True')
    parser.add_argument('--noise_question', type=str, default='False')
    parser.add_argument('--train_val_ratio', type=float, default=0.9)

    parser.add_argument('--prompt_mode', type=str, default='cat')
    parser.add_argument('--use_attention', type=bool, default=False)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=50)

    parser.add_argument("--save_path", type=str, default='./save/')

    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_path", type=str, default='./pretrained/base/base.pth')
    parser.add_argument("--data_path", type=str, default='./datasets/BindingDB/')

    parser.add_argument("--text_model_name", type=str, default='pubmed')
    parser.add_argument("--smiles_model_name", type=str, default='chembert')

    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--projection_layers', type=int, default=2)
    parser.add_argument('--num_questions', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=12,
                        help='input batch size for training (default: 32)')
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.0001, help='model learning rate for training (default: 0.00003)')

    parser.add_argument("--dataset", type=str,
                        choices=['toy_reg', 'BindingDB_cls', 'BindingDB_reg', 'JAK', 'AR', 'CYP_reg'],
                        default='BindingDB_cls')
    parser.add_argument("--protein_embedding", type=str, choices=['onehot', 'bio2vec', 'tape', 'esm2'], default='tape')
    parser.add_argument("--emb_size", type=int, choices=[20, 100, 768, 1280], default=768)
    parser.add_argument("--load_method", type=str, choices=['pkl', 'lmdb'], default='pkl')
    parser.add_argument("--train_text_model", type=str, choices=['True', 'False'], default='False')
    args = parser.parse_args()

    # Convert boolean string arguments to actual boolean
    args.train_kpgt = args.train_kpgt == 'True'
    args.phar_loss = args.phar_loss == 'True'
    args.align_loss = args.align_loss == 'True'
    args.noise_question = args.noise_question == 'True'
    args.use_norm_reg = args.use_norm_reg == 'True'
    args.train_text_model = args.train_text_model == 'True'
    return args


import torch.nn.functional as F


def attention_alignment_loss(attention_source, attention_target):
    cos_sim = F.cosine_similarity(attention_source, attention_target, dim=-1)
    return (1.0 - cos_sim).mean()


def get_question_embeddings(args):
    text_path = os.path.join(args.data_path, '../text', 'phar_question_howmany_gpt4o_27.json')

    with open(text_path, 'r', encoding='utf-8') as fp:
        text_list = json.load(fp)
    # Initialize lists for selected questions and pharmacophore names
    select_question = []
    select_discription = []
    phar_question_name = []
    if args.noise_question:
        select_question = ["To be, or not to be, that is the question."]
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
    question_texts, question_masks = text_model.tokenize(select_question, max_length=64)
    discription_texts, discription_mask = text_model.tokenize(select_discription, max_length=96)

    question_text_embeddings = text_model(input_ids=question_texts, attention_mask=question_masks)
    discription_text_embeddings = text_model(input_ids=discription_texts, attention_mask=discription_mask)

    text_dict = {
        'question_text_embeddings': question_text_embeddings.to(args.device),
        'discription_text_embeddings': discription_text_embeddings.to(args.device),
        'question_masks': question_masks.to(args.device),
        'discription_mask': discription_mask.to(args.device),
        'phar_question_name': phar_question_name
    }
    return text_dict


def load_file(path, filename, filetype='txt', encoding=None):
    if filename:
        filepath = os.path.join(path, filename)
    else:
        filepath = path
    if filetype == 'json':
        with open(filepath, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    elif filetype == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding=encoding)
    elif filetype == 'cPickle':
        with open(filepath, 'rb') as f:
            return cPickle.load(f)
    else:  # Default to text
        with open(filepath, 'r') as f:
            return f.read()


def process_compounds(ligands):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True) for d in ligands]


def build_smile_graph(compound_smiles, smile_file_name):
    if os.path.exists(smile_file_name):
        with open(smile_file_name, 'rb') as f:
            return pickle.load(f)
    else:
        smile_graph = {smile: smile_to_graph(smile) for smile in tqdm(compound_smiles)}
        with open(smile_file_name, 'wb+') as f:
            pickle.dump(smile_graph, f)
        return smile_graph


def load_features(paths):
    features = {}
    for key, path in paths.items():
        if key == 'fps':
            features[key] = torch.from_numpy(sps.load_npz(path).todense().astype(np.float32))
        elif key == 'mds':
            mds = np.load(path)['md'].astype(np.float32)
            features[key] = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
        elif key in ['phars', 'smiles_embeddings']:
            if path.endswith('.pkl'):
                features[key] = load_file(path, '', filetype='cPickle')
            else:
                env = lmdb.open(path, max_readers=1, readonly=True,
                                lock=False, readahead=False, meminit=False)
                with env.begin(write=False) as txn:
                    features[key] = list(txn.cursor().iternext(values=False))
        elif path.endswith('.pkl'):
            features[key] = load_file(path, '', filetype='cPickle')
    return features


def prepare_fold_entries(rows, cols, fold, smiles, target_keys, affinity, graphs, fps, mds, phars,
                         smiles_embeddings):
    
    row_indices = rows[fold]
    col_indices = cols[fold]

    smiles_batch = np.array(smiles)[row_indices] 
    target_keys_batch = np.array(target_keys)[col_indices]
    affinity_batch = affinity[row_indices, col_indices] 
    graphs_batch = np.array(graphs)[row_indices] 
    
    fps_batch = fps[row_indices] 
    mds_batch = mds[row_indices] 
    phars_batch = [f'{str(item)}'.encode() for item in row_indices] 
    smiles_embeddings_batch = phars_batch 

    fold_entries = {'compound_iso_smiles': smiles_batch,
                    'target_key': target_keys_batch,
                    'affinity': affinity_batch,
                    'graphs': graphs_batch,
                    'fps': fps_batch,
                    'mds': mds_batch,
                    'phars_idx': phars_batch,
                    'smiles_embeddings_idx': smiles_embeddings_batch}
    return fold_entries

def load_dataset_dta_CPI(args, question_name, evaluate=False):
    """
    load CPI dataset, support train/valid and ind_test  .
    return: train_data, valid_data, test_data
    """
    import os
    import numpy as np

    def get_paths(is_test):
        if not is_test:
            base_path = f"./datasets/BindingDB/{args.dataset}/train/"
        else:
            base_path = f"./datasets/BindingDB/{args.dataset}/ind_test/"
        return {
            'base': base_path,
            'smile_file': os.path.join(base_path, 'smile_graph'),
            'cache': os.path.join(base_path, f"{args.dataset}_5.pkl"),
            'fps': os.path.join(base_path, "rdkfp1-7_512.npz"),
            'mds': os.path.join(base_path, "molecular_descriptors.npz"),
            'phars': os.path.join(base_path, "phar_features.pkl"),
            'smiles_embeddings': os.path.join(base_path, f"smiles_embeddings_{args.smiles_model_name}.pkl"),
        }

    def load_entries_and_features(paths):
        fold = eval(load_file(paths['base'], 'valid_entries.txt'))
        ligands = load_file(paths['base'], 'compounds.txt', filetype='json')
        proteins = load_file(paths['base'], 'proteins.txt', filetype='json')
        affinity = load_file(paths['base'], 'Y', filetype='pickle', encoding='latin1')
        features = load_features({
            'fps': paths['fps'],
            'mds': paths['mds'],
            'phars': paths['phars'],
            'smiles_embeddings': paths['smiles_embeddings']
        })
        compound_smiles = process_compounds(ligands)
        smile_graph = build_smile_graph(compound_smiles, paths['smile_file'])
        graphs, _ = load_graphs(paths['cache'])
        phars_idx = features['phars']
        smiles_embeddings_idx = features['smiles_embeddings']
        fps = features['fps']
        mds = features['mds']
        target_graph = {key: target_matrics(key, args.embedding_path) for key in proteins}
        return fold, compound_smiles, proteins, affinity, graphs, fps, mds, phars_idx, smiles_embeddings_idx, smile_graph, target_graph

    def make_dataset(entries, compound_smiles, proteins, affinity, graphs, fps, mds, phars_idx, smiles_embeddings_idx, smile_graph, target_graph, question_name):
        rows, cols, fold = entries
        fold_entries = prepare_fold_entries(
            rows, cols, fold, compound_smiles, list(proteins.keys()), affinity, graphs, fps, mds, phars_idx, smiles_embeddings_idx
        )
        return pharmaQADataset_CPI(
            smiles=np.array(fold_entries['compound_iso_smiles']),
            proteins_keys=np.array(fold_entries['target_key']),
            targets=np.array(fold_entries['affinity']),
            smile_graph=smile_graph,
            smile_line_graph=fold_entries['graphs'],
            protein_graph=target_graph,
            fps=fold_entries['fps'],
            mds=fold_entries['mds'],
            phars=fold_entries['phars_idx'],
            smiles_dict=fold_entries['smiles_embeddings_idx'],
            phar_path=paths['phars'],
            smiles_embedding_path=paths['smiles_embeddings'],
            phar_question_name=question_name,
            question_num=args.num_questions,
            device=args.device,
            dataset_type=args.dataset_type
        )

    if not evaluate:
        paths = get_paths(is_test=False)
        print('Loading train/valid features from cache...')
        fold, compound_smiles, proteins, affinity, graphs, fps, mds, phars_idx, smiles_embeddings_idx, smile_graph, target_graph = load_entries_and_features(paths)
        np.random.seed(args.seed)
        np.random.shuffle(fold)
        split_idx = int(args.train_val_ratio * len(fold))
        train_fold, valid_fold = fold[:split_idx], fold[split_idx:]
        rows, cols = np.where(~np.isnan(affinity))
        print('Preparing train/valid fold entries...')
        train_data = make_dataset((rows, cols, train_fold), compound_smiles, proteins, affinity, graphs, fps, mds, phars_idx, smiles_embeddings_idx, smile_graph, target_graph, (rows, cols, train_fold), question_name)
        valid_data = make_dataset((rows, cols, valid_fold), compound_smiles, proteins, affinity, graphs, fps, mds, phars_idx, smiles_embeddings_idx, smile_graph, target_graph, (rows, cols, valid_fold), question_name)
        return train_data, valid_data, None
    else:
        paths = get_paths(is_test=True)
        print('Loading test features from cache...')
        fold, compound_smiles, proteins, affinity, graphs, fps, mds, phars_idx, smiles_embeddings_idx, smile_graph, target_graph = load_entries_and_features(paths)
        rows, cols = np.where(~np.isnan(affinity))
        print('Preparing test fold entries...')
        test_data = make_dataset((rows, cols, fold), compound_smiles, proteins, affinity, graphs, fps, mds, phars_idx, smiles_embeddings_idx, smile_graph, target_graph, question_name)
        return None, None, test_data
    
def train(model, device, train_loader, mean, std, text_dict, optimizer, loss_fn, epoch, writer):
    model.train()
    align_loss = 0
    train_loss = []
    phar_loss_fn = nn.MSELoss(reduction="none")
    align_loss_fn = attention_alignment_loss
    for batch_idx, data in enumerate(tqdm(train_loader)):

        (smiles, proteins, protein_lens, labels, \
         smiles_graph, g, ecfp, md, phar_targets, phar_target_mx, smiles_embedding, smiles_mask) = data

        proteins, protein_lens, ecfp, md, g, smiles_graph, labels, phar_targets = [
            x.to(device) for x in
            [proteins, protein_lens, ecfp, md, g, smiles_graph, labels, phar_targets]
        ]
        smiles_embedding, smiles_mask = smiles_embedding.to(device), smiles_mask.to(device)
        phar_target_mx = phar_target_mx.to(device)

        optimizer.zero_grad()

        output, pred_phar_num, atten, atten_protein_ligand, _ = model(ecfp, md, g, proteins, protein_lens, text_dict,
                                                                      smiles_embedding, smiles_mask)

        phar_y = phar_targets.to(torch.float32)

        if mean is not None and std is not None:
            output = ((output * std) + mean)

        pre_loss = loss_fn(output, labels.float())

        if args.phar_loss:
            phar_loss = phar_loss_fn(pred_phar_num, phar_y).mean()
        else:
            phar_loss = 0

        if args.align_loss:

            align_loss = (align_loss_fn(atten.view(atten.size()[0], -1),
                                        phar_target_mx.view(phar_target_mx.size()[0], -1))).mean()
        else:
            align_loss = 0

        loss = pre_loss + args.alpha * phar_loss + args.beta * align_loss

        loss.backward()
        optimizer.step()

        writer.add_scalar('Train/pre_loss', pre_loss, (epoch) * len(train_loader) + batch_idx + 1)
        writer.add_scalar('Train/align_loss', align_loss, (epoch) * len(train_loader) + batch_idx + 1)
        writer.add_scalar('Train/phar_loss', phar_loss, (epoch) * len(train_loader) + batch_idx + 1)
        writer.add_scalar('Train/loss_total', loss, (epoch) * len(train_loader) + batch_idx + 1)

    train_loss.append(loss.item())
    train_loss = np.average(train_loss)
    writer.add_scalar('Train/Loss', train_loss, epoch)
    return train_loss


def evaluate(model, device, loader, text_dict, loss_fn):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    eval_loss = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            (smiles, proteins, protein_lens, labels, \
             smiles_graph, g, ecfp, md, phar_targets, phar_target_mx, smiles_embedding, smiles_mask) = data

            proteins, protein_lens, ecfp, md, g, labels, smiles_graph, phar_targets = [
                x.to(device) for x in
                [proteins, protein_lens, ecfp, md, g, labels, smiles_graph, phar_targets]
            ]
            smiles_embedding, smiles_mask = smiles_embedding.to(device), smiles_mask.to(device)

            output, pred_phar_num, atten, atten_protein_ligand,_ = model(ecfp, md, g, proteins, protein_lens,
                                                                       text_dict, smiles_embedding, smiles_mask)

            total_preds = torch.cat((total_preds, output.to(torch.float32).cpu()), 0)
            total_labels = torch.cat((total_labels, labels.view(output.shape).to(torch.float32).cpu()), 0)

            predict_loss = loss_fn(output, labels)
        eval_loss.append(predict_loss.item())
    eval_loss = np.average(eval_loss)
    return eval_loss, total_labels.numpy().flatten(), total_preds.numpy().flatten()


def test(model, device, loader, text_dict):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            (smiles, proteins, protein_lens, labels, \
             smiles_graph, g, ecfp, md, phar_targets, phar_target_mx, smiles_embedding, smiles_mask) = data

            proteins, protein_lens, ecfp, md, g, smiles_graph, phar_targets = [
                x.to(device) for x in
                [proteins, protein_lens, ecfp, md, g, smiles_graph, phar_targets]
            ]
            smiles_embedding, smiles_mask = smiles_embedding.to(device), smiles_mask.to(device)

            output, pred_phar_num, atten, atten_protein_ligand,_ = model(ecfp, md, g, proteins, protein_lens,
                                                                       text_dict, smiles_embedding, smiles_mask)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.view(output.shape).to(torch.float64).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def check_update(data_type, predict_score, best_score):
    if data_type == 'regression' or data_type == 'loss':
        if predict_score <= best_score:
            return True
        else:
            return False
    elif data_type == 'classification':
        if predict_score >= best_score:
            return True
        else:
            return False


def get_score(dataset_type, T, P):
    if dataset_type == 'classification':
        AUC = roc_auc_score(T, P)
        tpr, fpr, _ = precision_recall_curve(T, P)
        AUPR = auc(fpr, tpr)
        return AUC, AUPR
    else:
        mse = get_mse(T, P)
        pearson = get_pearson(T, P)
        r2 = r2_score(T, P)
        return mse, pearson, r2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(args):
    LR = args.lr
    NUM_EPOCHS = args.n_epochs

    dataset = args.dataset
    TRAIN_BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.batch_size

    if dataset == 'BindingDB_reg':
        args.dataset_type = 'regression'
    elif dataset == 'BindingDB_cls':
        args.dataset_type = 'classification'
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    protein_embedding = args.protein_embedding
    emb_size = args.emb_size
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    parmeter = f'ratio_{args.train_val_ratio}_' + f'bach_{TRAIN_BATCH_SIZE}_' + f'LR_{LR}_' \
               + f'seed_{args.seed}_' + protein_embedding + f'_use_encoder_{args.use_encoder}'

    model_file_dir = '../save/' + dataset + f'/{formatted_datetime}/'

    if args.load_model_path:
        model_file_dir = '/'.join(args.load_model_path.split('/')[:-1])

    if not os.path.exists(model_file_dir):
        os.makedirs(model_file_dir)

    args.embedding_path = "./datasets/BindingDB/%s/%s/" % (dataset, protein_embedding)

    max_length = max(1500, int(open("./datasets/BindingDB/" + dataset + '/max_length.txt', 'r').read()))
    model_name = os.path.join(model_file_dir, parmeter + '.pt')
    log_dir = model_file_dir + f'/logs/{parmeter}/'
    writer = SummaryWriter(log_dir=model_file_dir + f'/logs/')
    if not os.path.exists(model_file_dir):
        os.makedirs(model_file_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # load data
    text_dict = get_question_embeddings(args)
    collator = Collator_pharVQA_CPI(config_dict['base']['path_length'])
    train_data, valid_data, _ = load_dataset_dta_CPI(args, text_dict['phar_question_name'])
    _, _, test_data = load_dataset_dta_CPI(args, text_dict['phar_question_name'], evaluate=True)

    # train_data = test_data
    # valid_data = test_data
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collator, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False,
                                               collate_fn=collator)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collator)
    
    # instantiate a model
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = pharmaPrompt_CPI(args, device, config_dict, 512, 200, vocab,
                            emb_size=emb_size, max_length=max_length, dropout=dropout_global,
                            train_dataset=train_data)

    if args.load_model_path:
        model_weight = torch.load(args.load_model_path, map_location='cpu')
        model.load_state_dict(model_weight)
        print(f'load model success on checkpoint {args.load_model_path}')

    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters()) / 1e6))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

 

    # train
    if args.dataset_type == 'classification':
        best_score = best_auc
        loss_fn = BCEWithLogitsLoss()
        mean = None
        std = None
    else:
        best_score = best_mse
        loss_fn = torch.nn.MSELoss()
        if args.use_norm_reg:
            mean = train_data.mean[0].to(device)
            std = train_data.std[0].to(device)
        else:
            mean = None
            std = None
    stop_epoch = 0
    best_epoch = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, device, train_loader, mean, std, text_dict, optimizer, loss_fn, epoch + 1, writer)
        val_loss, T, P = evaluate(model, device, valid_loader, text_dict, loss_fn)
        if mean is not None and std is not None:
            P = ((P * std.detach().cpu().numpy()) + mean.detach().cpu().numpy())

        valid_score = get_score(args.dataset_type, T, P)

        if args.dataset_type == 'classification':
            print(dataset, f'Valid at {epoch + 1} step \t auc score :', valid_score[0], 'aupr score', valid_score[1])
            writer.add_scalar('Valid/auc', valid_score[0], epoch)
            writer.add_scalar('Valid/aupr', valid_score[1], epoch)
        elif args.dataset_type == 'regression':
            print(dataset, f'Valid at {epoch + 1} step \t mse score :', valid_score[0], 'pearson score', valid_score[1],
                  'r2 score', valid_score[2])
            writer.add_scalar('Valid/mse', valid_score[0], epoch)
            writer.add_scalar('Valid/pearson', valid_score[1], epoch)
            writer.add_scalar('Valid/r2', valid_score[2], epoch)

        writer.add_scalar('Valid/Loss', val_loss, epoch)
        print('epoch\t', epoch + 1, 'train_loss\t', train_loss, 'val_loss\t', val_loss)

        stop_epoch += 1
        if check_update(args.dataset_type, valid_score[0], best_score):
            best_score = valid_score[0]
            stop_epoch = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_name)

            T, P = test(model, device, test_loader, text_dict)

            if mean is not None and std is not None:
                P = ((P * std.detach().cpu().numpy()) + mean.detach().cpu().numpy())

            test_score = get_score(args.dataset_type, T, P)
            if args.dataset_type == 'classification':
                predict = sigmoid(P)
                print(dataset, f'Test at {epoch + 1} step \t auc score :', test_score[0], 'aupr score', test_score[1])
                writer.add_scalar('Test/auc', test_score[0], epoch)
                writer.add_scalar('Test/aupr', test_score[1], epoch)
                np.savetxt(f'{log_dir}/Results_epoch_{epoch}.txt',
                           np.concatenate(
                               (T.reshape(-1, 1), predict.reshape(-1, 1), (predict >= 0.5).astype(int).reshape(-1, 1)),
                               axis=1), fmt='%.4f')
            elif args.dataset_type == 'regression':
                print(dataset, f'Test at {epoch + 1} step \t mse score :', test_score[0], 'pearson score',
                      test_score[1],
                      'r2 score', test_score[2])
                writer.add_scalar('Test/mse', test_score[0], epoch)
                writer.add_scalar('Test/pearson', test_score[1], epoch)
                writer.add_scalar('Test/r2', test_score[2], epoch)
                np.savetxt(f'{log_dir}/Results_epoch_{epoch}.txt',
                           np.concatenate((T.reshape(-1, 1), P.reshape(-1, 1)),
                                          axis=1), fmt='%.4f')
        if stop_epoch == early_stop:
            print('(EARLY STOP) No improvement since epoch ', best_epoch)
            break
        scheduler.step(val_loss)
    if args.dataset_type == 'classification':
        print(
            'Best epoch %s; best_test_auc_score%s; best_test_aupr_score%s; dataset:%s; train ratio:%s' % (
                best_epoch, test_score[0], test_score[1], dataset, args.train_val_ratio))
    elif args.dataset_type == 'regression':
        print(
            'Best epoch %s; best_test_mse_score%s; best_test_pearson_score%s; dataset:%s; train ratio:%s' % (
                best_epoch, test_score[0], test_score[1], dataset, args.train_val_ratio))


if __name__ == '__main__':
    args = parse_args()
    main(args)
