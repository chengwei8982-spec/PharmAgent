import logging
import os
import random
import numpy as np
import torch
import dgl
from torch import nn
from sklearn.metrics import r2_score
from rdkit import Chem
import networkx as nx

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25




def set_random_seed(seed=22, n_threads=16):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean', apply_logsoftmax=True, ignore_index=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.apply_logsoftmax = apply_logsoftmax
        self.ignore_idx = ignore_index

    def forward(self, logits, label):
        if logits.shape != label.shape and self.ignore_idx != -1:
            logits = logits[label != self.ignore_idx]
            label = label[label != self.ignore_idx]

        # Apply Label Smoothing:
        with torch.no_grad():
            if logits.shape != label.shape:
                new_label = torch.zeros(logits.shape)
                indices = torch.Tensor([[torch.arange(len(label))[i].item(),
                                         label[i].item()] for i in range(len(label))]).long()
                value = torch.ones(indices.shape[0])
                label = new_label.index_put_(tuple(indices.t()), value).to(label.device)
                label = label * (1 - self.smoothing) + self.smoothing / logits.shape[-1]
                label = label / label.sum(-1)[:, None]

            elif self.ignore_idx != -1:  # for context alignment loss
                label_lengths = (label != 2).sum(dim=-1)
                valid_indices = label_lengths != 0

                exist_align = (label == 1).sum(dim=-1) > 0
                smoothed_logits_addon = self.smoothing / label_lengths
                smoothed_logits_addon[smoothed_logits_addon > 1] = 0

                tmp = label.clone()
                tmp = tmp * (1 - self.smoothing) + smoothed_logits_addon.unsqueeze(1)
                tmp[label == 2] = 0

                label = tmp[valid_indices & exist_align]
                logits = logits[valid_indices & exist_align]

            else:
                label = label * (1 - self.smoothing) + self.smoothing / logits.shape[-1]
                label = label / label.sum(-1)[:, None]

        if self.apply_logsoftmax:
            logs = self.log_softmax(logits)
        else:
            logs = logits

        loss = -torch.sum(logs * label, dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features

def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

def _eval_rmse(y_true, y_pred, mean=None, std=None):
    '''
        compute RMSE score averaged across tasks
    '''
    rmse_list = []
    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        if (mean is not None) and (std is not None):
            rmse_list.append(np.sqrt(
                ((y_true[is_labeled, i] - (y_pred[is_labeled, i] * std[i] + mean[i])) ** 2).mean()))
        else:
            rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))
    return sum(rmse_list) / len(rmse_list)


def _eval_r2(y_true, y_pred, mean=None, std=None):
    '''
        compute R2 score averaged across tasks
    '''
    r2_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        if (mean is not None) and (std is not None):
            r2_list.append(r2_score(y_true[is_labeled, i], y_pred[is_labeled, i] * std[i] + mean[i]))
        else:
            r2_list.append(r2_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    return sum(r2_list) / len(r2_list)

def target_matrics(key, embedding_path):
    eembedding_file = os.path.join(embedding_path, key)
    input_feature = torch.load(eembedding_file)
    return input_feature['feature'], input_feature['size']

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index
