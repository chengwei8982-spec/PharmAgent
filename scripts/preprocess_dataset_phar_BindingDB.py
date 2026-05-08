import argparse
import json
import multiprocessing
import os
import pickle
import sqlite3
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool
from threading import Timer
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from rdkit import RDConfig, Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from tqdm import tqdm
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.light import TextEncoder

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
factory_names = factory.GetFeatureDefs()

text_path = os.path.join('./datasets/text', 'phar_question_howmany_gpt4o_27.json')

with open(text_path, 'r', encoding='utf-8') as fp:
    text_list = json.load(fp)

phar_question_name = []
# Iterate through the text list and extract questions and descriptions
for category, items in text_list.items():
    for item in items:
        phar_question_name.append(item['type'])

text_model_name = 'chembert'
# text_model_name = 'scibert'
text_model = TextEncoder(model_name=f'{text_model_name}', load=True)


def extract_smiles_embeddings_scibert(smiles):
    length = len(smiles)
    smiles_texts, smiles_mask = text_model.tokenize([smiles], max_length=length if length < 512 else 512)

    smiles_embeddings = text_model(input_ids=smiles_texts.reshape(1, -1),
                                   attention_mask=smiles_mask.reshape(1, -1)).squeeze()

    return {'smiles_embeddings': smiles_embeddings,
            'smiles_mask': smiles_mask}


def extract_smiles_embeddings_chembert(smiles):
    idx_list, idx_mask, adj_mask, adj_matx = text_model.tokenizer.encode(smiles)
    smiles_embeddings = text_model.model(input=torch.tensor(idx_list, dtype=torch.long).unsqueeze(0),
                                         imask=torch.tensor(idx_mask, dtype=torch.bool).unsqueeze(0),
                                         amask=torch.tensor(adj_mask, dtype=torch.float).unsqueeze(0),
                                         amatx=torch.tensor(adj_matx, dtype=torch.float).unsqueeze(0),
                                         ).squeeze()

    return {'smiles_embeddings': smiles_embeddings,
            'smiles_mask': torch.tensor(idx_mask, dtype=torch.long)}


def extract_phar_feature_multi(data):
    m = Chem.MolFromSmiles(data)
    # get 2D corr
    AllChem.Compute2DCoords(m)
    feats = factory.GetFeaturesForMol(m)
    # for every molecules, find corresponding features
    fp_name = []
    fp = []
    num_nodes = m.GetNumAtoms()

    attn_phar_atom = torch.zeros(len(phar_question_name), len(feats))

    for phar_index, phar_task in enumerate(phar_question_name):
        # Set attention based on matching pharmacophore name
        matching_phars = [i.GetType() == phar_task for i in feats]
        attn_phar_atom[phar_index, matching_phars] = 1

    # Calculate the number of pharmacophore targets for each question
    phar_targets_num = attn_phar_atom.sum(dim=1).long()

    atom_phar_target_map = torch.zeros(num_nodes, len(phar_question_name))
    phar_target_mx = torch.zeros((len(m.GetBonds()), len(phar_question_name)))

    for feat in feats:
        ids = feat.GetAtomIds()
        fp_name.append(['.'.join((feat.GetFamily(), feat.GetType())), ids, list(feat.GetPos())])
        fp.append('.'.join((feat.GetFamily(), feat.GetType())))

        # get target pharma
        phar_index = phar_question_name.index(feat.GetType())
        atom_phar_target_map[ids, phar_index] = 1

    bonded_atoms = set()
    for bond_index, bond in enumerate(m.GetBonds()):
        begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
        for question_idx, question in enumerate(phar_question_name):
            phar_target_mx[bond_index, question_idx] = atom_phar_target_map[
                [begin_atom_id, end_atom_id], question_idx].sum(dim=0).to(torch.bool).to(torch.float32)

        bonded_atoms.add(begin_atom_id)
        bonded_atoms.add(end_atom_id)

    atom_feat_list = []
    for atom_id in range(num_nodes):
        if atom_id not in bonded_atoms:
            atom_phar = torch.zeros(len(phar_question_name))
            for question_idx, question in enumerate(phar_question_name):
                atom_phar[question_idx] = atom_phar_target_map[[atom_id], question_idx].sum(dim=0).to(torch.bool).to(
                    torch.float32)
            atom_feat_list.append(atom_phar)

    # zero_rows = torch.zeros(2, len(phar_question_name))

    if len(atom_feat_list) != 0:
        # phar_target_mx = torch.cat((phar_target_mx, zero_rows), dim=0)
        atom_feat_tensor = torch.stack(atom_feat_list, dim=0)
        phar_target_mx = torch.cat((phar_target_mx, atom_feat_tensor), dim=0)
        # phar_target_mx = torch.cat((phar_target_mx, zero_rows), dim=0)

    return {'fp_name': fp_name,
            'phar_targets_num': phar_targets_num,
            'atom_phar_target_map': atom_phar_target_map,
            'phar_target_mx': phar_target_mx}


def check_valid(reactant):
    valid_index = []
    for i, r in enumerate(reactant):
        if Chem.MolFromSmiles(r) is not None:
            valid_index.append(i)
    reactant = [reactant[i] for i in valid_index]
    return reactant


def split_sequence(word_dict, sequence, ngram):
    sequence = '_' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def process_dataset(data_name, datas, num_workers=None, multiprocessing=True,input='smiles_embedding',text_model_name='chembert'):
    """
    Process the dataset with multiprocessing.
    """
    def process_single_data(data):
        if input == 'smiles_embedding' and text_model_name == 'chembert':
            results = extract_smiles_embeddings_chembert(data)
        elif input == 'smiles_embedding' and text_model_name == 'scibert':
            results = extract_smiles_embeddings_scibert(data)
        elif input == 'phar':
            results = extract_phar_feature_multi(data)
        else:
            raise ValueError(f"Unknown input type: {input}")
        return results
    if multiprocessing:
        # Use multiprocessing for feature extraction
        with Pool(processes=num_workers) as pool:
            result = list(tqdm(pool.imap(process_single_data, datas), total=len(datas)))
    else:
        result = []

        for data in tqdm(datas):
            if input == 'smiles_embedding' and text_model_name == 'chembert':
                results = extract_smiles_embeddings_chembert(data)
            elif input == 'smiles_embedding' and text_model_name == 'scibert':
                results = extract_smiles_embeddings_scibert(data)
            elif input == 'phar':
                results = extract_phar_feature_multi(data)
            else:
                raise ValueError(f"Unknown input type: {input}")
            result.append(results)
    return result


def main(args):
    data_path = f"{args.root_path}/{args.dataset}/{args.mode}/"
    ligands = json.load(open(data_path + 'compounds.txt'), object_pairs_hook=OrderedDict)

    # load compounds
    smiles = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]))
        smiles.append(lg)
    
    if args.input == 'phar':    
        out_path = os.path.join(args.root_path, f"{args.dataset}/{args.mode}/phar_features.pkl")
    else:
        out_path = os.path.join(args.root_path, f"{args.dataset}/{args.mode}/smiles_embeddings_{text_model_name}.pkl")
    num_workers = multiprocessing.cpu_count() - 4
    print(f'processing {args.dataset} {args.mode} dataset on {args.root_path}...')
    result = process_dataset(args.dataset, smiles, num_workers=num_workers, multiprocessing=False,input=args.input,text_model_name=args.text_model_name)

    with open(out_path, 'wb') as file_handle:
        pickle.dump(result, file_handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BindingDB_cls')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--root_path", type=str, default='./datasets/BindingDB/')
    parser.add_argument("--input", type=str, default='smiles_embedding',
                        choices=['phar', 'smiles_embedding'])
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--text_model_name", type=str, default='chembert')
    args = parser.parse_args()

    if args.dataset:
        data_names = [args.dataset]
    else:
        # Get the list of datasets
        data_names = sorted(os.listdir(args.root_path))
    for data_name in data_names:
        args.dataset = data_name
        main(args)
