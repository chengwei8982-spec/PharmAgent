import argparse
import os
import sys
import dgl.backend as F
import pandas as pd
import torch
import numpy as np
import json
import tqdm

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from src.data.splitters import random_split, random_scaffold_split, scaffold_split, balanced_scaffold_split, \
    moleculeace_split


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Arguments for training pharmaVQA")
    parser.add_argument('--root_path', type=str, default='datasets/vs/')
    parser.add_argument('--use_split_method', type=str, default='random_scaffold_split')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


def load_smiles_data(dataset_path):
    """Load SMILES data."""
    try:
        smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
    except KeyError:
        try:
            smiles_list = pd.read_csv(dataset_path)['SMILES'].dropna().tolist()
        except KeyError:
            raise ValueError(f"Could not find a 'smiles' or 'SMILES' column in CSV: {dataset_path}")
    
    return smiles_list


def create_split_directory(root_path, dataset):
    """Create the split directory."""
    split_path = os.path.join(root_path, f"{dataset}/splits/")
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    return split_path


def save_split_indices(split_path, train_index, valid_index, test_index, filename):
    """Save split indices."""
    save_index = [train_index.numpy(), valid_index.numpy(), test_index.numpy()]
    merged_array = np.array(save_index)
    filepath = os.path.join(split_path, filename)
    np.save(filepath, merged_array)
    print(f"Split indices saved to: {filepath}")


def process_use_recommend_split(dataset, dataset_path, root_path, seed):
    """Process the recommended split method."""
    smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
    print(f'{dataset} has {len(smiles_list)} smiles')

    if dataset in ['tox21', 'toxcast', 'sider', 'clintox', 'pcba', 'muv', 'esol', 'lipo', 'freesolv']:
        train_index, valid_index, test_index = random_split(smiles_list, seed=seed)
    elif dataset in ['bace', 'bbbp', 'hiv']:
        train_index, valid_index, test_index = balanced_scaffold_split(smiles_list, balanced=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    split_path = create_split_directory(root_path, dataset)
    save_split_indices(split_path, train_index, valid_index, test_index, "scaffold-MoleculeNet.npy")


def process_random_scaffold_split(dataset, dataset_path, root_path, seeds):
    """Process random scaffold splitting."""
    smiles_list = load_smiles_data(dataset_path)
    print(f'{dataset} has {len(smiles_list)} smiles')
    
    split_path = create_split_directory(root_path, dataset)
    
    for seed in seeds:
        train_index, valid_index, test_index = random_scaffold_split(
            torch.arange(len(smiles_list)), smiles_list=smiles_list, null_value=0, 
            frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
        )
        
        save_split_indices(split_path, train_index, valid_index, test_index, f"scaffold-{seed}.npy")


def process_common_scaffold_split(dataset, dataset_path, root_path, use_split_method):
    """Process the common scaffold split method."""
    smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
    print(f'{dataset} has {len(smiles_list)} smiles')
    
    train_index, valid_index, test_index = scaffold_split(
        torch.arange(len(smiles_list)), smiles_list=smiles_list, null_value=0, 
        frac_train=0.8, frac_valid=0.1, frac_test=0.1
    )
    
    split_path = create_split_directory(root_path, dataset)
    save_split_indices(split_path, train_index, valid_index, test_index, f"scaffold-{use_split_method}.npy")


def process_stratified_scaffold_split(dataset, dataset_path, root_path, use_split_method):
    """Process stratified scaffold splitting."""
    smile_df = pd.read_csv(dataset_path).dropna()
    smiles_list = smile_df['smiles'].tolist()
    print(f'{dataset} has {len(smiles_list)} smiles')
    
    task_names = smile_df.columns.drop(['smiles']).tolist()
    labels = smile_df[task_names].values
    
    train_index, valid_index, test_index = moleculeace_split(
        smiles_list, labels, val_size=0.1, test_size=0.1
    )
    
    split_path = create_split_directory(root_path, dataset)
    save_split_indices(split_path, train_index, valid_index, test_index, f"scaffold-{use_split_method}.npy")


def process_dataset(dataset, root_path, use_split_method, seeds):
    """Process a single dataset."""
    print(f"\nStart processing dataset: {dataset}")
    
    # Skip datasets that are not needed
    if dataset in ['DrugBank']:
        print(f"Skip dataset: {dataset}")
        return
    
    dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
    
    # Check whether the file exists
    if not os.path.exists(dataset_path):
        print(f"Warning: file does not exist: {dataset_path}")
        return
    
    try:
        if use_split_method == 'use_recommend_split':
            process_use_recommend_split(dataset, dataset_path, root_path, seeds[0])
        elif use_split_method == 'random_scaffold_split':
            process_random_scaffold_split(dataset, dataset_path, root_path, seeds)
        elif use_split_method == 'commn':
            process_common_scaffold_split(dataset, dataset_path, root_path, use_split_method)
        elif use_split_method == 'stratified_scaffold_split':
            process_stratified_scaffold_split(dataset, dataset_path, root_path, use_split_method)
        elif use_split_method == 'balanced':
            smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
            print(f'{dataset} has {len(smiles_list)} smiles')
            # TODO: Implement the balanced split method
        else:
            print(f"Warning: unknown split method: {use_split_method}")
            return
            
        print(f"Finished split processing for dataset {dataset}")
        
    except Exception as e:
        print(f"Error while processing dataset {dataset}: {str(e)}")


def main():
    """Main entry point."""
    args = parse_args()
    root_path = f'{path}/{args.root_path}'
    
    # Determine which datasets to process
    if args.dataset:
        data_list = [args.dataset]
    else:
        data_list = sorted(os.listdir(root_path))
    
    use_split_method = args.use_split_method
    seeds = [args.seed]
    
    print(f"Root path: {root_path}")
    print(f"Split method: {use_split_method}")
    print(f"Seeds: {seeds}")
    print(f"Dataset list: {data_list}")
    
    # Process each dataset
    for dataset in data_list:
        process_dataset(dataset, root_path, use_split_method, seeds)
    
    print(f"\nAll datasets have been processed!")


if __name__ == '__main__':
    main()
