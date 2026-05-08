from collections import OrderedDict
import json
import os
import pickle
import sys

from rdkit.Chem import AllChem, ChemicalFeatures
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem, RDLogger, RDConfig
from scipy import sparse as sp
import argparse

from src.data.featurizer import smiles_to_graph_tune, smiles_to_graph_phar
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

RDLogger.DisableLog('rdApp.*')

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, default='./datasets/moleculeace')
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args


def preprocess_dataset(args):
    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    df = df.dropna(axis=0, how='all')
    cache_file_path = f"{args.data_path}/{args.dataset}/{args.dataset}_{args.path_length}.pkl"
    smiless = df.smiles.values.tolist()
    task_names = df.columns.drop(['smiles']).tolist()
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                  smiless,
                  max_length=args.path_length,
                  n_virtual_nodes=2,
                  n_jobs=args.n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print('saving graphs')
    save_graphs(cache_file_path, valid_graphs,
                labels={'labels': labels})

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"{args.data_path}/{args.dataset}/rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{args.data_path}/{args.dataset}/molecular_descriptors.npz", md=arr[:, 1:])

def preprocess_dataset_ligand(args):
    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    df = df.dropna(axis=0, how='all')
    cache_file_path = f"{args.data_path}/{args.dataset}/{args.dataset}_{args.path_length}.pkl"
    try:
        smiless = df.smiles.values.tolist()
        task_names = df.columns.drop(['smiles']).tolist()
    except:
        smiless = df.Smiles.values.tolist()
        task_names = df.columns.drop(['Smiles']).tolist()
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                  smiless,
                  max_length=args.path_length,
                  n_virtual_nodes=2,
                  n_jobs=args.n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print('saving graphs')
    save_graphs(cache_file_path, valid_graphs,
                labels={'labels': labels})

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"{args.data_path}/{args.dataset}/rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(args.n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{args.data_path}/{args.dataset}/molecular_descriptors.npz", md=arr[:, 1:])

def split_sequence(word_dict, sequence, ngram):
    sequence = '_' + sequence + '='
    words = [word_dict[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]
    return np.array(words)

def preprocess_dataset_BindingDB(args):

    data_path = f"{args.data_path}/{args.dataset}/{args.mode}/"
    cache_file_path = f"{args.data_path}/{args.dataset}/{args.mode}/{args.dataset}_{args.path_length}"

    ligands = json.load(open(data_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    # proteins = json.load(open(data_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    # affinity = pickle.load(open(data_path + 'Y', 'rb'), encoding='latin1')
    # smile_file_name = data_path + '/smile_graph'

    # load compounds graph
    smiles = []
    smile_graph = {}
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        smiles.append(lg)
    # # load seqs
    # target_key = []
    # target_graph = {}
    # for key in proteins.keys():
    #     target_key.append(key)
    #     target_graph[key] = target_matrics(key, args.embedding_path)

    # Molecules pre-process
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                  smiles,
                  max_length=args.path_length,
                  n_virtual_nodes=2,
                  n_jobs=args.n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    valid_smiles = [smiles[idx] for idx in valid_ids]

    print('saving graphs')
    save_graphs(cache_file_path + '.pkl', valid_graphs,
                labels={'valid_idx': torch.LongTensor(valid_ids)})

    print('extracting reactant fingerprints')
    FP_list = []
    for r_smiles in valid_smiles:
        mol = Chem.MolFromSmiles(r_smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"{args.data_path}/{args.dataset}/{args.mode}/rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    fn = RDKit2DNormalized()
    features_map = []
    for mol in tqdm(valid_smiles):
        features_map.append(fn.process(mol))
    # generator = RDKit2DNormalized()
    # features_map = Pool(args.n_jobs).imap(generator.process, molecules)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{args.data_path}/{args.dataset}/{args.mode}/molecular_descriptors.npz", md=arr[:, 1:])

if __name__ == '__main__':
    args = parse_args()
    data_name = os.listdir(args.data_path)
    if args.dataset:
        print('processing {}'.format(args.dataset))
        if args.data_path.split('/')[-1] == 'BindingDB':
            for mode in ['train', 'ind_test']:
                args.mode = mode
                preprocess_dataset_BindingDB(args)
        else:
            preprocess_dataset(args)
    else:
        for dataset in data_name:
            print('processing {}'.format(dataset))
            args.dataset = dataset
            if args.data_path.split('/')[-1] == 'BindingDB':
                for mode in ['train', 'ind_test']:
                    args.mode = mode
                    preprocess_dataset_BindingDB(args)
            else:
                preprocess_dataset(args)
