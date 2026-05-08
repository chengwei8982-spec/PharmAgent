config_dict = {
    'base': {
        'd_node_feats': 137, 'd_edge_feats': 14, 'd_g_feats': 768,
        'd_hpath_ratio': 12, 'n_mol_layers': 12, 'path_length': 5,
        'n_heads': 12, 'n_ffn_dense_layers': 2, 'input_drop': 0.0, 'attn_drop': 0.1, 'feat_drop': 0.1,
        'batch_size': 32, 'lr': 2e-04, 'weight_decay': 1e-6,
        'candi_rate': 0.5, 'fp_disturb_rate': 0.5, 'md_disturb_rate': 0.5,
        'dropout': 0
    },
    'gnn': {
        'n_layers': 3,
        'hidden_dim': 512,
        'dropout_ratio': 0.3,
        'gnn_type': 'gin',
        'JK': 'last',
        'graph_pooling': 'mean'

    },
    'BAN': {'hidden_dim': 768,
            'out_dim': 768
            },
    'text_encoder': {
        'scibert': {
            'out_dim': 768
        },
        'pubmed': {
            'out_dim': 768
        },
        'molT5': {
            'out_dim': 512
        },
        'chembert': {
            'out_dim': 1024
        }
    },
    'smiles_encoder':
        {
            'scibert': {
                'out_dim': 768
            },
            'chembert': {
                'out_dim': 1024
            },
        },
    'head': {
        'out_dim': 768,
    }
}
molmcl_config = {
    'num_layer': 5,
    'emb_dim': 300,
    'JK': 'last',
    'drop_ratio': 0.0,
    'atom_feat_dim': 170,
    'bond_feat_dim': 14
}