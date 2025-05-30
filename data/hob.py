import os.path as osp
import pathlib
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
import pickle

import dgl
from dgl.data.utils import load_graphs, save_graphs
from dgl import backend as F


from.featurizer import smiles_to_graph, Vocab, N_BOND_TYPES, N_ATOM_TYPES



class HobDataset(object):
    def __init__(self, data_path, split, tasks,
                 metrics=['acc', 'f1_score', 'roc_auc', 'auprc'],
                 load=True):

        self.all_tasks = ['HOB']
        self.tasks = tasks
        self.task_type = 'classification'
        self.data_path = data_path
        self.split = split
        self.load = load
        self.metrics = metrics
        self.vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        self.preprocessed = osp.exists(osp.join(self.data_path, f"{self.split}.bin"))

        self._load()


    def _load(self):
        self._load_data()
        self._weight_balancing()

        num = len(self.smiles_list)
        if num >= 20000:
            self.batch_size = 128
        elif 20000 > num >= 10000:
            self.batch_size = 64
        elif 10000 > num >= 3000:
            self.batch_size = 32
        elif 3000 > num:
            self.batch_size = 16
        else:
            raise NotImplementedError(f'batch size not defined for {num} data')
        print(len(self.smiles_list), 'loaded!')


    def _load_data(self):
        if self.load and self.preprocessed:
            self.data_list, data_dict = load_graphs(osp.join(self.data_path, f"{self.split}.bin"))
            # 检查是否有可用的GPU
            if torch.cuda.is_available():
                device = 'cuda:0'  # 如果有，使用第一个GPU
            else:
                device = 'cpu'  # 如果没有，使用CPU

            # 将数据移到GPU上
            self.data_list = [g.to(device) for g in self.data_list]

            all_label_list, all_mask_list = data_dict['labels'], data_dict['masks']

            with open(osp.join(self.data_path, f'{self.split}_smiles.txt'), 'r') as f:
                smiles_ = f.readlines()
                smiles_list = [s.strip() for s in smiles_]

            with open(osp.join(self.data_path, f"{self.split}_node_feats.pkl"), 'rb') as f:
                self.node_feats = pickle.load(f)


        else:
            print('preprocessing data...')
            data_file = pathlib.Path(self.data_path, f"{self.split}.csv")
            all_data = pd.read_csv(data_file, usecols=['Smiles'] + self.all_tasks)

            smiless = all_data['Smiles'].values.tolist()
            targets = all_data[self.all_tasks]
            self.data_list, all_label_list, smiles_list, all_mask_list, length_list = [], [], [], [], []
            self.node_feats = []

            for smiles, label in zip(smiless, targets.iterrows()):

                mol = Chem.MolFromSmiles(smiles)
                cano_smiles = Chem.MolToSmiles(mol)
                length = F.tensor(np.array(len(cano_smiles)).astype(np.int64))

                data, node_feats = smiles_to_graph(cano_smiles, self.vocab)

                if torch.cuda.is_available:
                    data = data.to('cuda:0')  # move to GPU
                else:
                    data = data.to('cpu')  # move to CPU

                label = np.array(label[1].tolist())
                mask = np.ones_like(label)
                mask[np.isnan(label)] = 0  # 使用布尔索引，选取 label 中所有对应为 True 的位置的值（即 NaN 值）
                mask = F.tensor(mask.astype(np.float32))
                label[np.isnan(label)] = 0
                label = F.tensor(np.array(label.astype(np.float32)))

                self.data_list.append(data)
                all_label_list.append(label)
                all_mask_list.append(mask)
                smiles_list.append(cano_smiles)
                length_list.append(length)

                self.node_feats.append(node_feats)

            all_label_list = F.stack(all_label_list, dim=0)
            all_mask_list = F.stack(all_mask_list, dim=0)
            self.length_list = torch.stack(length_list)

            save_graphs(osp.join(self.data_path, f"{self.split}.bin"),
                        self.data_list,
                        labels={'labels': all_label_list,
                                'masks': all_mask_list,
                                })

            with open(osp.join(self.data_path, f"{self.split}_node_feats.pkl"), 'wb') as f:
                pickle.dump(self.node_feats, f)

            with open(osp.join(self.data_path, f"{self.split}_smiles.txt"), 'w') as f:
                for smiles in smiles_list:
                    f.write(smiles + '\n')

        label_list, mask_list = [], []
        for task in self.tasks:
            label_list.append(all_label_list[:, self.all_tasks.index(task)])
            mask_list.append(all_mask_list[:, self.all_tasks.index(task)])

        self.smiles_list = np.array(smiles_list)
        self.label_list = torch.stack(label_list, dim=-1)
        self.mask_list = torch.stack(mask_list, dim=-1)
        if len(self.tasks) == 1:
            remain = (self.mask_list == 1.0).squeeze(-1)
            self.label_list = self.label_list[remain]
            self.smiles_list = self.smiles_list[remain.numpy() == 1]
            self.data_list = np.array(self.data_list)[remain.numpy() == 1].tolist()
            self.mask_list = torch.ones_like(self.label_list)

        self.kpgt_feats = np.load(osp.join(self.data_path, f'kpgt_{self.split}.npz'))
        self.kpgt_feats = self.kpgt_feats['fps'].tolist()

    def _weight_balancing(self):
        num_pos = F.sum(self.label_list, dim=0)  # 计算正样本的数量
        num_indices = F.sum(self.mask_list, dim=0)  # 计算总样本数量
        self._task_pos_weights = (num_indices - num_pos) / num_pos  # 计算权重

    @property
    def task_pos_weights(self):
        return self._task_pos_weights

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, item):
        return (self.smiles_list[item], self.data_list[item], self.label_list[item], self.mask_list[item],
                self.node_feats[item], self.kpgt_feats[item])


