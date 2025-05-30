import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

import sys
sys.path.append("..")

from data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES

from data.finetune_dataset import MoleculeDataset
from data.load_data import collate_tune
from model.light import LiGhTPredictor as LiGhT



config_dict = {
    'base': {
        'd_node_feats': 137, 'd_edge_feats': 14, 'd_g_feats': 768, 'd_hpath_ratio': 12, 'n_mol_layers': 12, 'path_length': 5, 'n_heads': 12, 'n_ffn_dense_layers': 2,'input_drop':0.0, 'attn_drop': 0.1, 'feat_drop': 0.1, 'batch_size': 1024, 'lr': 2e-04, 'weight_decay': 1e-6,
        'candi_rate':0.5, 'fp_disturb_rate': 0.5, 'md_disturb_rate': 0.5
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--config", type=str, required=True, default=config_dict)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    return args


def extract_features(config, model_path, data_path, dataset, task, load=True):
    config = config['base']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    collator = collate_tune(config['path_length'])

    mol_dataset = MoleculeDataset(root_path=data_path, dataset=dataset, task=task, dataset_type=None)
    loader = DataLoader(mol_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size
        ).to(device)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path).items()})
    fps_list = []
    for batch_idx, batched_data in enumerate(loader):
        _, g, ecfp, md, labels = batched_data
        ecfp = ecfp.to(device)
        md = md.to(device)
        g = g.to(device)
        fps = model.generate_fps(g, ecfp, md)
        fps_list.extend(fps.detach().cpu().numpy().tolist())
    # np.savez_compressed(f"{data_path}/kpgt_{dataset}.npz", fps=np.array(fps_list))
    # print(f"The extracted features were saving at {data_path}/kpgt_{dataset}.npz")
    np.savez_compressed(f"{data_path}/kpgt_predict.npz", fps_list=fps_list)
    print(f"The extracted features were saving at {data_path}/kpgt_predict.npz")



if __name__ == '__main__':
    # set_random_seed(22,1)
    # args = parse_args()
    config = config_dict
    # dataset_split = ['train', 'test', 'val']
    # for dataset in dataset_split:
    #     data_path = '../dataset/logD3'
    #     model_path = '../saved_models/kpgt_pretrained/base/base.pth'
    #     task = 'logD'
    #     extract_features(config, model_path, data_path, dataset, task)
    data_path = '../dataset/probe_data/zinc500/probe_data'
    model_path = '../saved_models/kpgt_pretrained/base/base.pth'
    task = ''

    dataset = 'probe_data'

    extract_features(config, model_path, data_path, dataset, task)