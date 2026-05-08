import argparse
import concurrent
import json
import multiprocessing
import os
import pickle
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

import lmdb
import numpy as np
import pandas as pd
import torch
from rdkit import RDConfig, Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from tqdm import tqdm

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


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset_base_path", type=str, default='./datasets/benchmark')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--importance", type=str, default='')
    parser.add_argument("--input", type=str, default='smiles_embedding',
                        choices=['phar', 'smiles_embedding', 'smiles_input'])
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--multiprocessing", type=str, default="False")
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()
    args.multiprocessing = args.multiprocessing == 'True'
    return args


def extract_phar_feature_multi(data):
    """
    Process a single molecule to extract features.
    """
    # Example feature extraction logic from original function
    m = Chem.MolFromSmiles(data)
    AllChem.Compute2DCoords(m)

    feats = factory.GetFeaturesForMol(m)

    # Initialize variables
    fp_name = []
    fp = []
    num_nodes = m.GetNumAtoms()

    attn_phar_atom = torch.zeros(len(phar_question_name), len(feats))
    for phar_index, phar_task in enumerate(phar_question_name):
        matching_phars = [i.GetType() == phar_task for i in feats]
        attn_phar_atom[phar_index, matching_phars] = 1

    phar_targets_num = attn_phar_atom.sum(dim=1).long()
    atom_phar_target_map = torch.zeros(num_nodes, len(phar_question_name))
    phar_target_mx = torch.zeros((len(m.GetBonds()), len(phar_question_name)))

    for feat in feats:
        ids = feat.GetAtomIds()
        fp_name.append(['.'.join((feat.GetFamily(), feat.GetType())), ids, list(feat.GetPos())])
        fp.append('.'.join((feat.GetFamily(), feat.GetType())))

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

    if len(atom_feat_list) != 0:
        atom_feat_tensor = torch.stack(atom_feat_list, dim=0)
        phar_target_mx = torch.cat((phar_target_mx, atom_feat_tensor), dim=0)

    return {'fp_name': fp_name,
            'phar_targets_num': phar_targets_num,
            'atom_phar_target_map': atom_phar_target_map,
            'phar_target_mx': phar_target_mx}


def parallel_process_dataset(datas, num_workers=4):
    """
    Parallelize the process of feature extraction across multiple workers.
    """
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(extract_phar_feature_multi, datas), total=len(datas)))

    return results


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


def extract_smiles_token_chembert(smiles):
    idx_list, idx_mask, adj_mask, adj_matx = text_model.tokenizer.encode(smiles)

    return {'idx_list': torch.tensor(idx_list, dtype=torch.long).unsqueeze(0),
            'idx_mask': torch.tensor(idx_mask, dtype=torch.bool).unsqueeze(0),
            'adj_mask': torch.tensor(adj_mask, dtype=torch.float).unsqueeze(0),
            'adj_matx': torch.tensor(adj_matx, dtype=torch.float).unsqueeze(0),
            }


def check_valid(smiles):
    valid_index = []
    for i, p in enumerate(smiles):
        if Chem.MolFromSmiles(p) is not None:
            valid_index.append(i)
    product = [smiles[i] for i in valid_index]
    return product


def read_data(data_name, type=None):
    """
    Read the SMILES data from the respective file based on the dataset name.
    """
    if type:
        data_path = os.path.join(args.dataset_base_path, f'{data_name}/{type}.csv')
        datas = pd.read_csv(data_path)['SMILES']
        datas = datas.dropna(axis=0, how='all')
        return datas
    else:
        if data_name == 'chembl29':
            data_path = os.path.join(args.dataset_base_path, f'{data_name}/smiles.smi')
            with open(data_path, 'r') as f:
                lines = f.readlines()
                # Filter SMILES strings with length less than 300
                datas = [line.strip('\n') for line in lines if len(line.strip('\n')) < 300]
            datas = check_valid(datas)
        else:
            data_path = os.path.join(args.dataset_base_path, f'{data_name}/{data_name}.csv')
            try:
                datas = pd.read_csv(data_path)['smiles']
            except:
                datas = pd.read_csv(data_path)['SMILES']
            datas = datas.dropna(axis=0, how='all')

            datas = check_valid(datas.values)

        return datas


import logging

logging.basicConfig(level=logging.INFO)


def save_to_lmdb(result, data_name, dataset_base_path, text_model_name, type='', input_type='smiles_embedding'):
    """
    Function to save processed results to LMDB.
    """
    if input_type == 'smiles_embedding':
        out_path = os.path.join(dataset_base_path,
                                f"{data_name}/{type}/smiles_embeddings_{text_model_name}_lmdb" if type else f"{data_name}/smiles_embeddings_{text_model_name}_lmdb")
    elif input_type == 'phar':
        out_path = os.path.join(dataset_base_path,
                                f"{data_name}/{type}/phar_features_lmdb" if type else f"{data_name}/phar_features_lmdb")
    elif input_type == 'smiles_input':
        out_path = os.path.join(dataset_base_path,
                                f"{data_name}/{type}/smiles_input_{text_model_name}_lmdb" if type else f"{data_name}/smiles_input_{text_model_name}_lmdb")

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Folder {out_path} created.")
    else:
        print(f"Folder {out_path} already exists.")

    logging.info(f'Saving data to {out_path}...')

    env = lmdb.open(out_path, map_size=100 * 1024 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, res in tqdm(enumerate(result), total=len(result)):
            if res is not None:
                p_key = '{}'.format(i)
                try:
                    txn.put(p_key.encode(), pickle.dumps(res))
                except Exception as e:
                    logging.error(f"Error processing index {i} and product {p_key}: {str(e)}")
            else:
                logging.warning(f"Warning. Process failed at index {i}")


def process_dataset(data_name, datas, dataset_base_path, type='', use_multiprocessing=True, input='smiles_embedding',
                    num_workers=4):
    """
    Main function to process the dataset with optional parallelization.
    """
    logging.info(f'Processing {data_name} {type} dataset...')

    def process_single_data(data):
        """Process a single data item based on the input type."""
        if input == 'smiles_embedding' and text_model_name == 'chembert':
            return extract_smiles_embeddings_chembert(data)
        elif input == 'smiles_embedding' and text_model_name == 'scibert':
            return extract_smiles_embeddings_scibert(data)
        elif input == 'phar':
            return extract_phar_feature_multi(data)
        else:
            raise ValueError(f"Unknown input type: {input}")

    result = [process_single_data(data) for data in tqdm(datas)]

    # Save results to LMDB
    save_to_lmdb(result, data_name, dataset_base_path, text_model_name, type, input)


def main(args):
    if args.dataset_base_path.split('/')[-1] == 'benchmark':
        for data_name in args.data_names:
            if '.' in data_name:
                continue
            else:
                # Read data based on dataset type
                datas = read_data(data_name)
                process_dataset(data_name, datas, args.dataset_base_path, use_multiprocessing=args.multiprocessing,
                                type='', input=args.input)

    elif args.dataset_base_path.split('/')[-1] in ('vs', '_server_cache'):
        for data_name in args.data_names:
            datas = read_data(data_name)
            process_dataset(data_name, datas, args.dataset_base_path, type='',
                            use_multiprocessing=args.multiprocessing,
                            input=args.input)
    elif args.dataset_base_path.split('/')[-1] == 'ligand':
        for data_name in args.data_names:
            for type in ['train', 'ind_test']:

                args.type = type

                data_path = f"{args.dataset_base_path}/{data_name}/{args.type}/"
                ligands = json.load(open(data_path + 'compounds.txt'), object_pairs_hook=OrderedDict)

                # load compounds
                datas = []
                for d in ligands.keys():
                    lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
                    datas.append(lg)

                process_dataset(data_name, datas, args.dataset_base_path, type,
                                use_multiprocessing=args.multiprocessing,
                                input=args.input)


if __name__ == '__main__':
    args = parse_args()
    if args.dataset:
        # Get the list of datasets
        args.data_names = [args.dataset]
    else:
        # Get the list of datasets
        args.data_names = sorted(os.listdir(args.dataset_base_path))

    # Define the number of workers for multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main(args)
