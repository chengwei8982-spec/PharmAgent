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
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Arguments for training pharmaVQA")
    parser.add_argument('--root_path', type=str, default='datasets/vs/')
    parser.add_argument('--use_split_method', type=str, default='random_scaffold_split')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


def load_smiles_data(dataset_path):
    """加载SMILES数据"""
    try:
        smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
    except KeyError:
        try:
            smiles_list = pd.read_csv(dataset_path)['SMILES'].dropna().tolist()
        except KeyError:
            raise ValueError(f"CSV文件中没有找到'smiles'或'SMILES'列: {dataset_path}")
    
    return smiles_list


def create_split_directory(root_path, dataset):
    """创建分割目录"""
    split_path = os.path.join(root_path, f"{dataset}/splits/")
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    return split_path


def save_split_indices(split_path, train_index, valid_index, test_index, filename):
    """保存分割索引"""
    save_index = [train_index.numpy(), valid_index.numpy(), test_index.numpy()]
    merged_array = np.array(save_index)
    filepath = os.path.join(split_path, filename)
    np.save(filepath, merged_array)
    print(f"分割索引已保存到: {filepath}")


def process_use_recommend_split(dataset, dataset_path, root_path, seed):
    """处理推荐分割方法"""
    smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
    print(f'{dataset} has {len(smiles_list)} smiles')

    if dataset in ['tox21', 'toxcast', 'sider', 'clintox', 'pcba', 'muv', 'esol', 'lipo', 'freesolv']:
        train_index, valid_index, test_index = random_split(smiles_list, seed=seed)
    elif dataset in ['bace', 'bbbp', 'hiv']:
        train_index, valid_index, test_index = balanced_scaffold_split(smiles_list, balanced=True)
    else:
        raise ValueError(f"未知的数据集: {dataset}")

    split_path = create_split_directory(root_path, dataset)
    save_split_indices(split_path, train_index, valid_index, test_index, "scaffold-MoleculeNet.npy")


def process_random_scaffold_split(dataset, dataset_path, root_path, seeds):
    """处理随机骨架分割方法"""
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
    """处理通用骨架分割方法"""
    smiles_list = pd.read_csv(dataset_path)['smiles'].dropna().tolist()
    print(f'{dataset} has {len(smiles_list)} smiles')
    
    train_index, valid_index, test_index = scaffold_split(
        torch.arange(len(smiles_list)), smiles_list=smiles_list, null_value=0, 
        frac_train=0.8, frac_valid=0.1, frac_test=0.1
    )
    
    split_path = create_split_directory(root_path, dataset)
    save_split_indices(split_path, train_index, valid_index, test_index, f"scaffold-{use_split_method}.npy")


def process_stratified_scaffold_split(dataset, dataset_path, root_path, use_split_method):
    """处理分层骨架分割方法"""
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
    """处理单个数据集"""
    print(f"\n开始处理数据集: {dataset}")
    
    # 跳过不需要的数据集
    if dataset in ['DrugBank']:
        print(f"跳过数据集: {dataset}")
        return
    
    dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"警告: 文件不存在 {dataset_path}")
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
            # TODO: 实现balanced分割方法
        else:
            print(f"警告: 未知的分割方法: {use_split_method}")
            return
            
        print(f"完成数据集 {dataset} 的分割处理")
        
    except Exception as e:
        print(f"处理数据集 {dataset} 时出错: {str(e)}")


def main():
    """主函数"""
    args = parse_args()
    root_path = f'{path}/{args.root_path}'
    
    # 确定要处理的数据集列表
    if args.dataset:
        data_list = [args.dataset]
    else:
        data_list = sorted(os.listdir(root_path))
    
    use_split_method = args.use_split_method
    seeds = [args.seed]
    
    print(f"根路径: {root_path}")
    print(f"分割方法: {use_split_method}")
    print(f"种子: {seeds}")
    print(f"数据集列表: {data_list}")
    
    # 处理每个数据集
    for dataset in data_list:
        process_dataset(dataset, root_path, use_split_method, seeds)
    
    print(f"\n所有数据集处理完成!")


if __name__ == '__main__':
    main()
