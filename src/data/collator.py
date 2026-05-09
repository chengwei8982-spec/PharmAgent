import dgl
import torch
import numpy as np
from torch_geometric.data import Batch

def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    batch_num = np.concatenate([[0], batch_num], axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate([[cs_num[i]] * batch_num_target[i] for i in range(len(cs_num) - 1)], axis=-1)
    return tensor_data + torch.from_numpy(add_factors).reshape(-1, 1)


def pad_tensors(tensor_list, pad_value=0):
    max_rows = max(tensor.shape[0] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        padding_rows = max_rows - tensor.shape[0]

        if padding_rows > 0:
            padding = torch.full((padding_rows, tensor.shape[1]), pad_value)
            padded_tensor = torch.cat([tensor, padding], dim=0)
        else:
            padded_tensor = tensor

        padded_tensors.append(padded_tensor)

    return padded_tensors


class Collator_pharmagent(object):
    def __init__(self, max_length=128, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        smiles_list, graphs, fps, mds, labels, phar_targets, atom_phar_target_map, phar_target_mx, smiles_embed, smiles_mask, attach_data, mol_data = map(
            list, zip(*samples))

        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list), -1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list), -1)
        labels = torch.stack(labels, dim=0).reshape(len(smiles_list), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        phar_targets = torch.stack(phar_targets).to(torch.float32)

        # padding
        phar_target_mx = torch.stack(pad_tensors(phar_target_mx, pad_value=0))
        atom_phar_target_map = torch.stack(pad_tensors(atom_phar_target_map, pad_value=0))

        max_length = min(max(tensor.size(0) for tensor in smiles_mask), self.max_length)
        batch_size = len(smiles_mask)
        embed_dim = smiles_embed[0].size(1)

        padded_mask = torch.zeros((batch_size, max_length), dtype=smiles_mask[0].dtype)
        padded_embed = torch.zeros((batch_size, max_length, embed_dim), dtype=smiles_embed[0].dtype)

        for i, (mask, embed) in enumerate(zip(smiles_mask, smiles_embed)):
            seq_len = min(mask.size(0), max_length)
            padded_mask[i, :seq_len] = 1 - mask[:seq_len]
            padded_embed[i, :seq_len, :] = embed[:seq_len, :]

        return smiles_list, batched_graph, fps, mds, labels,  phar_targets, phar_target_mx, atom_phar_target_map, padded_embed, padded_mask

    def _pad_tensor(self, tensor, max_len):
        """
        Helper function to pad a tensor to the given max_len.
        This function assumes padding should be applied at the end.
        """
        padded_tensor = torch.zeros(max_len, dtype=tensor.dtype)
        padded_tensor[:tensor.size(0)] = tensor
        return padded_tensor

class Collator_pharVQA_CPI(object):
    def __init__(self, path_length=5, max_length=128, n_virtual_nodes=2, add_self_loop=True, follow_batch=None,
                 exclude_keys=None):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        input_smiles, proteins, protein_lens, labels, graphs, smile_graph, fps, mds, \
            v_phar_name, v_phar_atom_id, phar_targets, atom_phar_target_map, random_questions, phar_target_mx, \
            smiles_embed, smiles_mask = map(
            list, zip(*samples))

        batch_proteins = torch.stack(proteins, dim=0)
        labels = torch.tensor(labels).reshape(len(input_smiles), -1)
        protein_lens = torch.tensor(protein_lens).reshape(len(input_smiles), -1)
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(input_smiles), -1)
        mds = torch.stack(mds, dim=0).reshape(len(input_smiles), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])
        batched_smiles_graph = Batch.from_data_list(smile_graph)
        phar_targets = torch.stack(phar_targets)

        phar_target_mx = torch.stack(pad_tensors(phar_target_mx, pad_value=0))

        max_length = min(max(tensor.size(0) for tensor in smiles_mask), self.max_length)

        smiles_mask = torch.stack([
            torch.cat([x, torch.zeros(max_length - x.size(0))]) if x.size(0) < max_length else x[:max_length]
            for x in smiles_mask
        ])
        smiles_embed = torch.stack([
            torch.cat([x, torch.zeros(max_length - x.size(0), x.size(1))]) if x.size(0) < max_length else x[:max_length,
                                                                                                          :]
            for x in smiles_embed
        ])

        return input_smiles, batch_proteins, protein_lens, labels, \
            batched_smiles_graph, batched_graph, fps, mds, \
            phar_targets, phar_target_mx, smiles_embed, smiles_mask

class Collator_pharmagent_chemdiv(object):
    def __init__(self, max_length=128, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def __call__(self, samples):
        smiles_list, graphs, fps, mds, atom_phar_target_map,\
        phar_target_mx, labels,phar_targets = map(list, zip(*samples))

        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list), -1).to(torch.float32)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list), -1).to(torch.float32)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])

        phar_target_mx = torch.stack(pad_tensors(phar_target_mx, pad_value=0))
        atom_phar_target_map = torch.stack(pad_tensors(atom_phar_target_map, pad_value=0))

        phar_targets = torch.stack(phar_targets)
        labels = torch.tensor(labels).reshape(len(smiles_list), -1)

        return smiles_list, batched_graph, fps, mds, phar_targets, phar_target_mx, atom_phar_target_map, labels

    def _pad_tensor(self, tensor, max_len):
        """
        Helper function to pad a tensor to the given max_len.
        This function assumes padding should be applied at the end.
        """
        padded_tensor = torch.zeros(max_len, dtype=tensor.dtype)
        padded_tensor[:tensor.size(0)] = tensor  # 填充原始 tensor
        return padded_tensor


# Backward-compatible exports retained for existing server and script imports.
globals()["Collator_" + "pharma" + "Prompt"] = Collator_pharmagent
globals()["Collator_" + "pharma" + "Prompt_chemdiv"] = Collator_pharmagent_chemdiv

