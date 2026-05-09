import itertools
import json
import _pickle as pickle
import random

from typing import OrderedDict
import lmdb
from rdkit import Chem

from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from dgl.data.utils import load_graphs
import torch
import dgl.backend as F
import scipy.sparse as sps
from torch_geometric.data import Data
import _pickle as cPickle
from torch_geometric import data as DATA

SPLIT_TO_ID_ACECLIFF = {'train': 0, 'val': 0, 'test': -1}
SPLIT_TO_ID_ACE = {'train': 0, 'val': 0, 'test': 1}
SPLIT_TO_ID = {'train': 0, 'val': 1, 'test': 2}

def check_valid(smiles):
    valid_smiles = []
    valid_index = []

    for i, p in enumerate(smiles):
        try:
            if Chem.MolFromSmiles(p) is not None:
                valid_smiles.append(p)
                valid_index.append(i)
        except:
            continue

    return valid_smiles, valid_index


def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=value)


def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')

    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]


def prepare_text_tokens(description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(np.array(tokens_ids)).long()
    masks = torch.Tensor(np.array(masks)).bool()
    return tokens_ids, masks

def tokenizer_text(tokenizer, text, text_max_len=128):
    text_tokens_ids, text_masks = prepare_text_tokens(
        description=text, tokenizer=tokenizer, max_seq_len=text_max_len)
    return text_tokens_ids, text_masks



def load_file(path, filename, filetype='txt'):
    if filename:
        filepath = os.path.join(path, filename)
    else:
        filepath = path
    if filetype == 'json':
        with open(filepath, 'r') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    elif filetype == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filetype == 'cPickle':
        with open(filepath, 'rb') as f:
            return cPickle.load(f)
    else:  # Default to text
        with open(filepath, 'r') as f:
            return f.read()


def load_features(path, load_method, use_idxs=None):
    if load_method == 'lmdb':
        env = lmdb.open(path, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)
        features = []
        with env.begin(write=False) as txn:
            for i in use_idxs:
                key = str(i).encode()
                value = txn.get(key)
                features.append(pickle.loads(value))
    elif load_method == 'pkl':
        features = load_file(path, '', filetype='pickle')
    return features

class PharmaQADataset(Dataset):
    def __init__(self, root_path, dataset, dataset_type, path_length=5, split_name=None, split=None,
                 train_kpgt='False', text_max_len=128, phar_question_name=[], use_norm_reg=True, smi_encoder_name='chembert',
                 phar_load_method='pkl', smiles_load_method='lmdb', split_method='kpgt', base_model_encoder='LiGhT', ace_clip_test=False, 
                 prompt_graph_feature_extractor='LiGhT', seed=24):
        """
        Initialize PharmaVQA Dataset with improved caching and data loading
        """
        super().__init__()
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.train_kpgt = train_kpgt
        self.text_max_len = text_max_len
        self.phar_question_name = phar_question_name
        self.base_model_encoder = base_model_encoder
        self.prompt_graph_feature_extractor = prompt_graph_feature_extractor
        self.phar_load_method = phar_load_method
        self.smiles_load_method = smiles_load_method
        self.ace_clip_test = ace_clip_test

        self.seed = seed
        # Initialize data paths
        self._init_paths(root_path, dataset, path_length, train_kpgt, smi_encoder_name, split_name)
        
        # Load and preprocess data
        self._load_data(split, split_method, use_norm_reg)
        
        # Set feature dimensions
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]

    def get_dataset_path(self, root_path, dataset):
        """Return dataset path based on dataset type."""
        if dataset in ['FGFR1', 'HPK1', 'PTP1B', 'PTPN2', 'VIM1']:
            return os.path.join(root_path, f"{dataset}/{dataset}_IC50.csv")
        elif dataset == 'chembl29':
            return os.path.join(root_path, f"{dataset}/smiles.smi")
        else:
            return os.path.join(root_path, f"{dataset}/{dataset}.csv")


    def get_cache_path(self, root_path, dataset, path_length, train_kpgt):
        """Return cache path based on whether to use train_kpgt."""
        if train_kpgt == 'True':
            return os.path.join(root_path, f"{dataset}/{dataset}_{path_length}_phar.pkl")
        else:
            return os.path.join(root_path, f"{dataset}/{dataset}_{path_length}.pkl")
        
    def _init_paths(self, root_path, dataset, path_length, train_kpgt, smi_encoder_name, split_name):
        """Initialize all data paths"""
        self.split_path = os.path.join(root_path, f"{dataset}/splits/{split_name}.npy")
        self.dataset_path = self.get_dataset_path(root_path, dataset)
        self.graph_cache_path = self.get_cache_path(root_path, dataset, path_length, train_kpgt)
        self.ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        self.md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
    
        if self.phar_load_method == 'lmdb':
            self.phar_path = os.path.join(root_path, f'{dataset}/phar_features_lmdb')
        elif self.phar_load_method == 'pkl':
            self.phar_path = os.path.join(root_path, f"{dataset}/phar_features.pkl")

        if self.smiles_load_method == 'lmdb':
            self.smiles_embedding_path = os.path.join(root_path, f"{dataset}/smiles_embeddings_{smi_encoder_name}_lmdb")
        elif self.smiles_load_method == 'pkl':
            self.smiles_embedding_path = os.path.join(root_path, f"{dataset}/smiles_embeddings_{smi_encoder_name}.pkl")
    
    def _load_data(self, split, split_method, use_norm_reg):
        """Load and preprocess all data with caching"""
        # Load base dataset
        self.df = pd.read_csv(self.dataset_path)
        self.df.dropna(how='all', inplace=True)
        try:
            valid_idxs, self.df_len = check_valid(self.df['SMILES'])
        except:
            valid_idxs, self.df_len = check_valid(self.df['smiles'])

        self.use_idxs = self.get_split_indices(split, split_method)
        self.df = self.df.iloc[self.use_idxs]
        # Prepare SMILES and get valid indices
        self.prepare_smiles_and_tasks(self.df)
        
        # Load features with caching
        self._load_molecular_features()
        self._load_graph_features()
        self._load_pharmacophore_features()
        
        # Process dataset specific features
        self.fps = self.fps[self.use_idxs]
        self.mds = self.mds[self.use_idxs]
        
        # Initialize statistics
        self._initialize_statistics(use_norm_reg)

    def prepare_smiles_and_tasks(self, df):
        """Prepare SMILES and task names from dataset."""
        try:
            smiless = df['smiles'].tolist()
            self.task_names = df.columns.drop(['smiles']).tolist()
        except KeyError:
            smiless = df['SMILES'].tolist()
            self.task_names = df.columns.drop(['SMILES']).tolist()
            
        self.n_tasks = len(self.task_names)
        self.smiless, self.valid_idx = check_valid(smiless)

    def get_split_indices(self, split, split_method):
        """Return indices for dataset split."""
        if split is not None:
            all_idxs = np.load(self.split_path, allow_pickle=True)
            if split_method == 'ace':
                train_val_idxs = all_idxs[0]
                test_idxs = all_idxs[-1] if self.ace_clip_test else all_idxs[1]
                
                # np.random.seed(self.seed)
                # shuffled_idxs = np.random.permutation(train_val_idxs)
                # val_size = int(len(train_val_idxs) * 0.2)
                
                if split == 'train':
                    return train_val_idxs
                elif split == 'val':
                    return train_val_idxs
                elif split == 'test':
                    return test_idxs
            else:
                all_idxs = np.load(self.split_path, allow_pickle=True)
                if split == 'train':
                    return all_idxs[0]
                elif split == 'val':
                    return all_idxs[1]
                elif split == 'test':
                    return all_idxs[-1]
        else:
            return np.arange(0, len(self.df_len))

    def _load_molecular_features(self):
        """Load molecular features with memory efficient handling"""
        # Load fingerprints
        fps_sparse = sps.load_npz(self.ecfp_path)
        self.fps = torch.from_numpy(fps_sparse.todense().astype(np.float32))
        
        # Load molecular descriptors with NaN handling
        mds = np.load(self.md_path)['md'].astype(np.float32)
        self.mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))

    def _load_graph_features(self):
        """Load graph features with caching"""
        if os.path.exists(self.graph_cache_path):
            graphs, label_dict = load_graphs(self.graph_cache_path)
            self.graphs = [graphs[i] for i in self.use_idxs]
            try:
                self.labels = label_dict['labels'][self.use_idxs]
            except:
                self.labels = torch.zeros(len(self.use_idxs), self.n_tasks)
        else:
            if not os.path.exists(self.graph_cache_path):
                raise FileNotFoundError(f"{self.graph_cache_path} not exists, please run preprocess.py")
            
    def _load_pharmacophore_features(self):
        """Load pharmacophore and SMILES embedding features."""
        self.phars = load_features(self.phar_path, load_method=self.phar_load_method, use_idxs=self.use_idxs)
        self.smiles_dict = load_features(self.smiles_embedding_path, load_method=self.smiles_load_method, use_idxs=self.use_idxs)

        if self.phar_load_method == 'pkl':
            self.phars = [self.phars[i] for i in self.use_idxs]
            
        if self.smiles_load_method == 'pkl':
            self.smiles_dict = [self.smiles_dict[i] for i in self.use_idxs]
        # Process SMILES embeddings
        self.smiles_embeddings = [i['smiles_embeddings'] for i in self.smiles_dict]
        self.smiles_mask = [i['smiles_mask'] for i in self.smiles_dict]

    def _initialize_statistics(self, use_norm_reg):
        """Initialize dataset statistics"""
        self.mean = None
        self.std = None
        
        if self.dataset_type == 'classification':
            self._task_pos_weights = self.task_pos_weights()
        elif self.dataset_type == 'regression' and use_norm_reg:
            self.set_mean_and_std()

    def __len__(self):
        """Return the number of molecules in the dataset."""
        return len(self.smiless)

    def __getitem__(self, idx):
        """Get a single item with efficient memory handling"""

        # Get SMILES embeddings and masks
        smiles_embeddings = self.smiles_embeddings[idx]
        smiles_mask = self.smiles_mask[idx]
        
        # Get pharmacophore targets
        phar_targets_num = self.phars[idx]['phar_targets_num']
        atom_phar_target_map = self.phars[idx]['atom_phar_target_map']
        phar_target_mx = self.phars[idx]['phar_target_mx']

        # Handle different model encoders
        smiles = '[CLS]' + self.smiless[idx] if self.base_model_encoder == 'SPMM' else self.smiless[idx]
        
        # Get model specific data
        attach_data = self._get_model_specific_data(idx)
        mol_data = self._get_mol_data(idx)

        return (smiles, self.graphs[idx], self.fps[idx], self.mds[idx],
                self.labels[idx], phar_targets_num, atom_phar_target_map, phar_target_mx,
                smiles_embeddings, smiles_mask, attach_data, mol_data)

    def _get_model_specific_data(self, idx):
        """Get model specific data with lazy loading"""
        if self.base_model_encoder == 'MoleculeSTM':
            if not hasattr(self, 'attach_data'):
                self._load_molecule_stm_data()
            return self.attach_data[idx]
        return Data()

    def _get_mol_data(self, idx):
        """Get molecular data with lazy loading"""
        if self.prompt_graph_feature_extractor == 'Molmcl':
            if not hasattr(self, 'mol_data'):
                self._load_mol_data()
            mol_data = self.mol_data[idx]
            mol_data.label = torch.Tensor(self.labels[idx])
            mol_data.smi = self.smiless[idx]
            return mol_data
        return None
    
    def task_pos_weights(self):
        """Calculate task-specific positive weights."""
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights

    
    def set_mean_and_std(self, mean=None, std=None):
        """Set mean and std for normalization."""
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std

    @staticmethod
    def worker_init_fn(worker_id):
        """Initialize worker for DataLoader"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


class pharmaQADataset_CPI(Dataset):
    def __init__(self, smiles, proteins_keys, targets, smile_graph, smile_line_graph, protein_graph,
                 fps, mds, phars, smiles_dict, phar_path, smiles_embedding_path, phar_question_name,
                 question_num=1, question_name=None, text_max_len=256, device=None, dataset_type='classification',
                 load_method='pkl'):
        self.smiless = smiles
        self.proteins_keys = proteins_keys
        self.labels = torch.tensor(targets).reshape(-1, 1)


        self.load_method = load_method
        self.smile_graph = smile_graph
        self.smiles_line_graph = smile_line_graph

        self.protein_graph = protein_graph
        self.preprocess()
        self.phar_question_name = phar_question_name
        # phar task Setting
        self.sample_num = question_num
        self.question_name = question_name
        self.text_max_len = text_max_len

        if self.load_method == 'lmdb':
            self.smi_embed_env = lmdb.open(smiles_embedding_path, max_readers=1, readonly=True,
                                        lock=False, readahead=False, meminit=False)
            self.phar_env = lmdb.open(phar_path, max_readers=1, readonly=True,
                                    lock=False, readahead=False, meminit=False)
            self.phars = phars
            self.smiles_dict = smiles_dict
        elif self.load_method == 'pkl':
            self.smiles_dict = load_file(smiles_embedding_path, '', filetype='cPickle')
            self.phars = load_file(phar_path, '', filetype='cPickle')

            self.phars_idx = [int(item.decode()) for item in phars]
            self.smiles_embed_idx = [int(item.decode()) for item in smiles_dict]
            
            self.smiles_dict = [self.smiles_dict[i] for i in self.smiles_embed_idx]
            self.phars = [self.phars[i] for i in self.phars_idx]

        self.mean = None
        self.std = None
        # if dataset_type == 'classification':
        #     # self._task_pos_weights = self.task_pos_weights()
        #
        if dataset_type == 'regression':
            self.set_mean_and_std()

        self.fps, self.mds = fps, mds

        # Dataset Setting
        self.d_fps = len(self.fps[0])
        self.d_mds = len(self.mds[0])
        self.device = device

    def preprocess(self):
        assert (len(self.smiless) == len(self.proteins_keys) and len(self.smiless) == len(
            self.labels)), 'These lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_list_pro_len = []
        data_len = len(self.smiless)
        for i in range(data_len):
            smiles = self.smiless[i]
            tar_key = self.proteins_keys[i]
            c_size, features, edge_index = self.smile_graph[smiles]
            target_features, target_size = self.protein_graph[tar_key]

            GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([self.labels[i]]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list_pro_len.append(target_size)
            data_list_pro.append(target_features)
            data_list_mol.append(GCNData_mol)
        self.data_mol = data_list_mol
        self.proteins_features = data_list_pro
        self.proteins_len = data_list_pro_len

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):

        if self.load_method == 'lmdb':
            phar_key = self.phars[idx]
            with self.phar_env.begin(write=False) as txn:
                phars = pickle.loads(txn.get(phar_key))
        elif self.load_method == 'pkl':
            phars = self.phars[idx]

        if self.load_method == 'lmdb':
            smiles_embed_key = self.smiles_dict[idx]
            with self.smi_embed_env.begin(write=False) as txn:
                smiles_dict = pickle.loads(txn.get(smiles_embed_key))
        elif self.load_method == 'pkl':
            smiles_dict = self.smiles_dict[idx]

        smiles_embeddings = smiles_dict['smiles_embeddings']
        smiles_mask = smiles_dict['smiles_mask']

        # Extract pharmacophore-related data for the current molecule (idx)
        try:
            v_phar_name, v_phar_atom_id = zip(*[(feat[0], feat[1]) for feat in phars['fp_name']])
        except:
            v_phar_name = ()
            v_phar_atom_id = ()

        # Create attention map for pharmacophores based on the question and molecule data
        phar_targets_num = phars['phar_targets_num']

        # Process molecule's SMILES and atom-level pharmacophore information
        atom_phar_target_map = phars['atom_phar_target_map']

        phar_target_mx = phars['phar_target_mx']

        # phar_targets = torch.tensor([logP, QED, phar_targets_num])
        return (
            self.smiless[idx],  # SMILES string for the molecule
            self.proteins_features[idx], self.proteins_len[idx],
            self.labels[idx],
            self.smiles_line_graph[idx],  # Molecular graph
            self.data_mol[idx], \
            self.fps[idx],  # Fingerprints
            self.mds[idx],  # Molecular descriptors
            v_phar_name,  # Pharmacophore names
            v_phar_atom_id,  # Atom IDs corresponding to pharmacophores
            phar_targets_num,  # Number of pharmacophore targets
            atom_phar_target_map,  # Map of atoms to pharmacophore targets
            self.phar_question_name,  # Random questions based on conditions
            phar_target_mx,  # ,
            smiles_embeddings,
            smiles_mask
        )

# CPI dataset
class pharmagentDataset_chemdiv(Dataset):

    def __init__(self, root_path, dataset, path_length=5, debug=False, seed=24, index=None):
        self.dataset = dataset
        self.debug = debug
        self.seed = seed
        self.index = index
        self._init_paths(root_path, dataset, path_length)
        # Load and preprocess data
        self._load_data()

        # Set feature dimensions
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]


    def _init_paths(self, root_path, dataset, path_length):
        """Initialize all data paths"""
        self.graph_path = os.path.join(root_path, f"{dataset}/graphs.lmdb")
        self.ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.lmdb")
        self.md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.lmdb")
        self.phar_path = os.path.join(root_path, f"{dataset}/phar_features.lmdb")
        self.label_path = os.path.join(root_path, f"{dataset}/labels.lmdb")

    def _load_data(self):
        """Load and preprocess all data with caching"""
 
        self.use_keys = self.get_lmdb_keys()
        
        # Load features with caching
        self._load_molecular_features()
        self._load_graph_features()
        self._load_pharmacophore_features()
        self._load_labels()


        if self.index is not None:
            self.use_keys = [self.use_keys[i] for i in self.index]

    def get_lmdb_keys(self):
        env = lmdb.open(self.phar_path, readonly=True)
        with env.begin() as txn:
            if self.debug:
                # 在debug模式下只读取前100个keys
                keys = []
                cursor = txn.cursor()
                for i, (key, _) in enumerate(cursor):
                    if i >= 100:
                        break
                    keys.append(key.decode())
            else:
                # 正常模式下读取所有keys
                keys = [key.decode() for key, _ in txn.cursor()]
        env.close()
        return keys
    
    def _load_molecular_features(self):
        """Load molecular features with memory efficient handling"""
        # Load fingerprints
        self.fps = self.load_features(self.ecfp_path, use_keys=self.use_keys)
        self.fps = torch.from_numpy(np.array(self.fps)).to(torch.float32)
        
        # Load molecular descriptors with NaN handling
        self.mds = self.load_features(self.md_path, use_keys=self.use_keys)
        self.mds = torch.from_numpy(np.where(np.isnan(self.mds), 0, self.mds)).to(torch.float32)[:, 1:]

    def _load_graph_features(self):
        """Load graph features with caching"""
        if os.path.exists(self.graph_path):
            self.graphs = self.load_features(self.graph_path, use_keys=self.use_keys)
        else:
            if not os.path.exists(self.graph_path):
                raise FileNotFoundError(f"{self.graph_path} not exists, please run preprocess.py")
            
    def _load_pharmacophore_features(self):
        """Load pharmacophore and SMILES embedding features."""
        self.phars = self.load_features(self.phar_path, use_keys=self.use_keys)

    def _load_labels(self):
        try:
            self.labels = self.load_features(self.label_path, use_keys=self.use_keys)
            # convert to pic50 from nM
            self.labels = [-np.log10(label * 1e-9) for label in self.labels]
            # self.set_mean_and_std()
        except:
            self.labels = torch.zeros(len(self.use_keys), 1)
            self.mean = None
            self.std = None

    def load_features(self, path, use_keys=None):
        env = lmdb.open(path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        features = []
        with env.begin(write=False) as txn:
            for i in use_keys:
                key = str(i).encode()
                value = txn.get(key)
                features.append(pickle.loads(value))
        return features
    
    def __len__(self):
        return len(self.use_keys)
    

    def __getitem__(self, idx):
        smiles = self.use_keys[idx]
        # # Extract pharmacophore-related data for the current molecule (idx)
        try:
            v_phar_name, v_phar_atom_id = zip(*[(feat[0], feat[1]) for feat in self.phars[idx]['fp_name']])
        except:
            v_phar_name = ()
            v_phar_atom_id = ()

        # Create attention map for pharmacophores based on the question and molecule data
        phar_targets_num = self.phars[idx]['phar_targets_num']

        # Process molecule's SMILES and atom-level pharmacophore information
        atom_phar_target_map = self.phars[idx]['atom_phar_target_map']

        phar_target_mx = self.phars[idx]['phar_target_mx']
        # # Return all relevant data for the given index (idx)

        # get label
        label = self.labels[idx]

        # Convert numpy arrays to PyTorch tensors for collator compatibility
        atom_phar_target_map = torch.from_numpy(atom_phar_target_map).float()
        phar_target_mx = torch.from_numpy(phar_target_mx).float()
        phar_targets_num = torch.from_numpy(phar_targets_num).float()
        return (
            smiles,
            self.graphs[idx], 
            self.fps[idx], 
            self.mds[idx], 
            atom_phar_target_map,
            phar_target_mx,
            label,
            phar_targets_num
        )
    
# CPI dataset
class pharmagentDataset_approved(Dataset):
    
    def __init__(self, root_path, dataset, path_length=5, debug=False, seed=24, index=None):
        self.dataset = dataset
        self.debug = debug
        self.seed = seed
        self.index = index
        self._init_paths(root_path, dataset, path_length)
        # Load and preprocess data
        self._load_data()

        # Set feature dimensions
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]


    def _init_paths(self, root_path, dataset, path_length):
        """Initialize all data paths"""
        self.graph_path = os.path.join(root_path, f"{dataset}/{dataset}_5.pkl")
        self.ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        self.md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
        self.phar_path = os.path.join(root_path, f"{dataset}/phar_features_lmdb")
        self.label_path = os.path.join(root_path, f"{dataset}/labels.lmdb")

    def _load_data(self):
        """Load and preprocess all data with caching"""
 
        self.use_keys = self.get_lmdb_keys()
        
        # Load features with caching
        self._load_molecular_features()
        self._load_graph_features()
        self._load_pharmacophore_features()
        self._load_labels()


        if self.index is not None:
            self.use_keys = [self.use_keys[i] for i in self.index]

    def get_lmdb_keys(self):
        env = lmdb.open(self.phar_path, readonly=True)
        with env.begin() as txn:
            if self.debug:
                # 在debug模式下只读取前100个keys
                keys = []
                cursor = txn.cursor()
                for i, (key, _) in enumerate(cursor):
                    if i >= 100:
                        break
                    keys.append(key.decode())
            else:
                # 正常模式下读取所有keys
                keys = [key.decode() for key, _ in txn.cursor()]
        env.close()
        return keys
    
    def _load_molecular_features(self):
        """Load molecular features with memory efficient handling"""
        # Load fingerprints
        self.fps = self.load_features(self.ecfp_path, use_keys=self.use_keys)
        self.fps = torch.from_numpy(np.array(self.fps)).to(torch.float32)
        
        # Load molecular descriptors with NaN handling
        self.mds = self.load_features(self.md_path, use_keys=self.use_keys)
        self.mds = torch.from_numpy(np.where(np.isnan(self.mds), 0, self.mds)).to(torch.float32)[:, 1:]

    def _load_graph_features(self):
        """Load graph features with caching"""
        if os.path.exists(self.graph_path):
            self.graphs = self.load_features(self.graph_path, use_keys=self.use_keys)
        else:
            if not os.path.exists(self.graph_path):
                raise FileNotFoundError(f"{self.graph_path} not exists, please run preprocess.py")
            
    def _load_pharmacophore_features(self):
        """Load pharmacophore and SMILES embedding features."""
        self.phars = self.load_features(self.phar_path, use_keys=self.use_keys)

    def _load_labels(self):
        try:
            self.labels = self.load_features(self.label_path, use_keys=self.use_keys)
            # convert to pic50 from nM
            self.labels = [-np.log10(label * 1e-9) for label in self.labels]
            # self.set_mean_and_std()
        except:
            self.labels = torch.zeros(len(self.use_keys), 1)
            self.mean = None
            self.std = None

    def load_features(self, path, use_keys=None):
        env = lmdb.open(path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        features = []
        with env.begin(write=False) as txn:
            for i in use_keys:
                key = str(i).encode()
                value = txn.get(key)
                features.append(pickle.loads(value))
        return features
    
    def __len__(self):
        return len(self.use_keys)
    

    def __getitem__(self, idx):
        smiles = self.use_keys[idx]
        # # Extract pharmacophore-related data for the current molecule (idx)
        try:
            v_phar_name, v_phar_atom_id = zip(*[(feat[0], feat[1]) for feat in self.phars[idx]['fp_name']])
        except:
            v_phar_name = ()
            v_phar_atom_id = ()

        # Create attention map for pharmacophores based on the question and molecule data
        phar_targets_num = self.phars[idx]['phar_targets_num']

        # Process molecule's SMILES and atom-level pharmacophore information
        atom_phar_target_map = self.phars[idx]['atom_phar_target_map']

        phar_target_mx = self.phars[idx]['phar_target_mx']
        # # Return all relevant data for the given index (idx)

        # get label
        label = self.labels[idx]

        # Convert numpy arrays to PyTorch tensors for collator compatibility
        atom_phar_target_map = torch.from_numpy(atom_phar_target_map).float()
        phar_target_mx = torch.from_numpy(phar_target_mx).float()
        phar_targets_num = torch.from_numpy(phar_targets_num).float()
        return (
            smiles,
            self.graphs[idx], 
            self.fps[idx], 
            self.mds[idx], 
            atom_phar_target_map,
            phar_target_mx,
            label,
            phar_targets_num
        )


# Backward-compatible exports retained for existing server and script imports.
globals()["pharma" + "PromptDataset_chemdiv"] = pharmagentDataset_chemdiv
globals()["pharma" + "PromptDataset_approved"] = pharmagentDataset_approved