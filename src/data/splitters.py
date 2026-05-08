import sys

import torch
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

from sklearn.cluster import SpectralClustering
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from typing import List
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from src.data.moleculeace import ActivityCliffs, get_tanimoto_matrix, RANDOM_SEED


# splitter function

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


# # test generate_scaffold
# s = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
# scaffold = generate_scaffold(s)
# assert scaffold == 'c1ccc(Oc2ccccn2)cc1'

def determined_scaffold_split(dataset, smiles_list, train_idx, valid_idx, test_idx,
                              return_smiles=False):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)


def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=False)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)


def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset


#
# def random_split(dataset, task_idx=None, null_value=0,
#                  frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0,
#                  smiles_list=None):
#     """
#
#     :param dataset:
#     :param task_idx:
#     :param null_value:
#     :param frac_train:
#     :param frac_valid:
#     :param frac_test:
#     :param seed:
#     :param smiles_list: list of smiles corresponding to the dataset obj, or None
#     :return: train, valid, test slices of the input dataset obj. If
#     smiles_list != None, also returns ([train_smiles_list],
#     [valid_smiles_list], [test_smiles_list])
#     """
#     np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
#
#     if task_idx != None:
#         # filter based on null values in task_idx
#         # get task array
#         y_task = np.array([data.y[task_idx].item() for data in dataset])
#         non_null = y_task != null_value  # boolean array that correspond to non null values
#         idx_array = np.where(non_null)[0]
#         dataset = dataset[torch.tensor(idx_array)]  # examples containing non
#         # null labels in the specified task_idx
#     else:
#         pass
#
#     num_mols = len(dataset)
#     random.seed(seed)
#     all_idx = list(range(num_mols))
#     random.shuffle(all_idx)
#
#     train_idx = all_idx[:int(frac_train * num_mols)]
#     valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
#                                                    + int(frac_train * num_mols)]
#     test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]
#
#     assert len(set(train_idx).intersection(set(valid_idx))) == 0
#     assert len(set(valid_idx).intersection(set(test_idx))) == 0
#     assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols
#
#     train_dataset = dataset[torch.tensor(train_idx)]
#     valid_dataset = dataset[torch.tensor(valid_idx)]
#     test_dataset = dataset[torch.tensor(test_idx)]
#
#     if not smiles_list:
#         return train_dataset, valid_dataset, test_dataset
#     else:
#         train_smiles = [smiles_list[i] for i in train_idx]
#         valid_smiles = [smiles_list[i] for i in valid_idx]
#         test_smiles = [smiles_list[i] for i in test_idx]
#         return train_dataset, valid_dataset, test_dataset, (train_smiles,
#                                                             valid_smiles,
#                                                             test_smiles)

def moleculeace_split(smiles: List[str], bioactivity: List[float], in_log10: bool = True, n_clusters: int = 5,
                         val_size: float = 0.1, test_size: float = 0.1, similarity: float = 0.9,
                         potency_fold: int = 10, remove_stereo: bool = False):
    """ Split data into train/test according to activity cliffs and compounds characteristics.
    :param smiles: (List[str]) list of SMILES strings
    :param bioactivity: (List[float]) list of bioactivity values
    :param in_log10: (bool) are the bioactivity values in log10?
    :param n_clusters: (int) number of clusters the data is split into for getting homogeneous data splits
    :param test_size: (float) test split
    :param similarity:  (float) similarity threshold for calculating activity cliffs
    :param potency_fold: (float) potency difference threshold for calculating activity cliffs
    :param remove_stereo: (bool) Remove racemic mixtures altogether?
    :return: df[smiles, exp_mean [nM], y, cliff_mol, split]
    """

    if remove_stereo:
        stereo_smiles_idx = [smiles.index(i) for i in find_stereochemical_siblings(smiles)]
        smiles = [smi for i, smi in enumerate(smiles) if i not in stereo_smiles_idx]
        bioactivity = [act for i, act in enumerate(bioactivity) if i not in stereo_smiles_idx]
        if len(stereo_smiles_idx) > 0:
            print(f"Removed {len(stereo_smiles_idx)} stereoisomers")

    if not in_log10:
        bioactivity = (-np.log10(bioactivity)).tolist()

    cliffs = ActivityCliffs(smiles, bioactivity)
    cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, similarity=similarity, potency_fold=potency_fold)

    # Perform spectral clustering on a tanimoto distance matrix
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_SEED, affinity='precomputed')
    clusters = spectral.fit(get_tanimoto_matrix(smiles)).labels_

    train_idx, test_idx, val_idx = [], [], []
    for cluster in range(n_clusters):

        cluster_idx = np.where(clusters == cluster)[0]
        clust_cliff_mols = [cliff_mols[i] for i in cluster_idx]

        # Can only split stratiefied on cliffs if there are at least 3 cliffs present, else do it randomly
        if sum(clust_cliff_mols) > 2:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size+val_size,
                                                               random_state=RANDOM_SEED,
                                                               stratify=clust_cliff_mols, shuffle=True)
        else:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size+val_size,
                                                               random_state=RANDOM_SEED,
                                                               shuffle=True)

        clust_test_idx, clust_val_idx = train_test_split(clust_test_idx, test_size=test_size/(test_size+val_size),
                                                         random_state=RANDOM_SEED, shuffle=True)

        train_idx.extend(clust_train_idx)
        val_idx.extend(clust_val_idx)
        test_idx.extend(clust_test_idx)

    return train_idx, val_idx, test_idx

def cv_random_split(dataset, fold_idx=0,
                    frac_train=0.9, frac_valid=0.1, seed=0,
                    smiles_list=None):
    """

    :param dataset:
    :param task_idx:
    :param null_value:
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed:
    :param smiles_list: list of smiles corresponding to the dataset obj, or None
    :return: train, valid, test slices of the input dataset obj. If
    smiles_list != None, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """

    np.testing.assert_almost_equal(frac_train + frac_valid, 1.0)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [data.y.item() for data in dataset]

    idx_list = []

    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, val_idx = idx_list[fold_idx]

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(val_idx)]

    return train_dataset, valid_dataset


from rdkit import Chem
from typing import List, Tuple, Union, Dict, Set


def smiles_to_scaffold(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def random_split(data: List[str],
                 sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed=0
                 ) -> Tuple[List[int], List[int], List[int]]:
    data_size = len(data)
    index_list = list(range(data_size))
    random.seed(seed)
    random.shuffle(index_list)

    train_size = int(sizes[0] * data_size)
    train_val_size = int((sizes[0] + sizes[1]) * data_size)

    train = index_list[:train_size]
    val = index_list[train_size:train_val_size]
    test = index_list[train_val_size:]

    return train, val, test


def balanced_scaffold_split(data: List[str],
                            sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                            balanced: bool = True) -> Tuple[List[int], List[int], List[int]]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = smiles_to_scaffold(data, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []

        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)

        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = list(scaffold_to_indices.values())
        random.shuffle(index_sets)
        index_sets = sorted(index_sets,
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
          f'train scaffolds = {train_scaffold_count:,} | '
          f'val scaffolds = {val_scaffold_count:,} | '
          f'test scaffolds = {test_scaffold_count:,}')

    return train, val, test


if __name__ == "__main__":
    from loader import MoleculeDataset
    from rdkit import Chem
    import pandas as pd

    # # test scaffold_split
    dataset = MoleculeDataset('dataset/tox21', dataset='tox21')
    smiles_list = pd.read_csv('dataset/tox21/processed/smiles.csv', header=None)[0].tolist()

    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                                                                frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    # train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = 0)
    unique_ids = set(train_dataset.data.id.tolist() +
                     valid_dataset.data.id.tolist() +
                     test_dataset.data.id.tolist())
    assert len(unique_ids) == len(dataset)  # check that we did not have any
    # missing or overlapping examples

    # test scaffold_split with smiles returned
    dataset = MoleculeDataset('dataset/bbbp', dataset='bbbp')
    smiles_list = pd.read_csv('dataset/bbbp/processed/smiles.csv', header=None)[
        0].tolist()
    train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles,
                                                 test_smiles) = \
        scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                       frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                       return_smiles=True)
    assert len(train_dataset) == len(train_smiles)
    for i in range(len(train_dataset)):
        data_obj_n_atoms = train_dataset[i].x.size()[0]
        smiles_n_atoms = len(list(Chem.MolFromSmiles(train_smiles[
                                                         i]).GetAtoms()))
        assert data_obj_n_atoms == smiles_n_atoms
    assert len(valid_dataset) == len(valid_smiles)
    for i in range(len(valid_dataset)):
        data_obj_n_atoms = valid_dataset[i].x.size()[0]
        smiles_n_atoms = len(list(Chem.MolFromSmiles(valid_smiles[
                                                         i]).GetAtoms()))
        assert data_obj_n_atoms == smiles_n_atoms
    assert len(test_dataset) == len(test_smiles)
    for i in range(len(test_dataset)):
        data_obj_n_atoms = test_dataset[i].x.size()[0]
        smiles_n_atoms = len(list(Chem.MolFromSmiles(test_smiles[
                                                         i]).GetAtoms()))
        assert data_obj_n_atoms == smiles_n_atoms

    # test random_split
    from loader import MoleculeDataset

    dataset = MoleculeDataset('dataset/tox21', dataset='tox21')
    train_dataset, valid_dataset, test_dataset = random_split(dataset, task_idx=None, null_value=0, frac_train=0.8,
                                                              frac_valid=0.1, frac_test=0.1)
    unique_ids = set(train_dataset.data.id.tolist() +
                     valid_dataset.data.id.tolist() +
                     test_dataset.data.id.tolist())
    assert len(unique_ids) == len(dataset)  # check that we did not have any
    # missing or overlapping examples

    # test random_split with smiles returned
    dataset = MoleculeDataset('dataset/bbbp', dataset='bbbp')
    smiles_list = pd.read_csv('dataset/bbbp/processed/smiles.csv', header=None)[
        0].tolist()
    train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles,
                                                 test_smiles) = \
        random_split(dataset, task_idx=None, null_value=0,
                     frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42,
                     smiles_list=smiles_list)
    assert len(train_dataset) == len(train_smiles)
    for i in range(len(train_dataset)):
        data_obj_n_atoms = train_dataset[i].x.size()[0]
        smiles_n_atoms = len(list(Chem.MolFromSmiles(train_smiles[
                                                         i]).GetAtoms()))
        assert data_obj_n_atoms == smiles_n_atoms
    assert len(valid_dataset) == len(valid_smiles)
    for i in range(len(valid_dataset)):
        data_obj_n_atoms = valid_dataset[i].x.size()[0]
        smiles_n_atoms = len(list(Chem.MolFromSmiles(valid_smiles[
                                                         i]).GetAtoms()))
        assert data_obj_n_atoms == smiles_n_atoms
    assert len(test_dataset) == len(test_smiles)
    for i in range(len(test_dataset)):
        data_obj_n_atoms = test_dataset[i].x.size()[0]
        smiles_n_atoms = len(list(Chem.MolFromSmiles(test_smiles[
                                                         i]).GetAtoms()))
        assert data_obj_n_atoms == smiles_n_atoms
