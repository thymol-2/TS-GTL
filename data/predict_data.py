import os
import os.path as osp

import numpy as np
import torch
from rdkit import Chem
import pickle
from dgl.data.utils import load_graphs, save_graphs
from.featurizer import smiles_to_graph, Vocab, N_BOND_TYPES, N_ATOM_TYPES

class PredictDataset(object):
    def __init__(self, data_path, metrics=['acc', 'f1_score', 'roc_auc', 'auprc'], load=True):
        self.data_path = data_path
        self.load = load
        self.metrics = metrics
        self.vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        self.preprocessed = os.path.exists(osp.join(self.data_path,
                                                    f"predict_data.bin"))
        self._load()

    def _load(self):
        self._load_data()
        num = len(self.smiles_list)
        if num >= 50000:
            self.batch_size = 512
        elif 50000 > num >= 30000:
            self.batch_size = 256
        elif 30000 > num >= 20000:
            self.batch_size = 128
        elif 20000 > num >= 10000:
            self.batch_size = 64
        elif 10000 > num >= 3000:
            self.batch_size = 32
        elif 3000 > num:
            self.batch_size = 16
        else:
            raise NotImplementedError(f'batch size not defined for {num} data ')
        print(len(self.smiles_list), "loaded!")


    def _load_data(self):
        if self.load and self.preprocessed:
            self.data_list, _ = load_graphs(osp.join(self.data_path,
                                                     f"predict_data.bin"))
            # 检查是否有可用的GPU
            if torch.cuda.is_available():
                device = 'cuda:0'  # 如果有，使用第一个GPU
            else:
                device = 'cpu'  # 如果没有，使用CPU
            # 将数据移到GPU上
            self.data_list = [g.to(device) for g in self.data_list]

            with open(osp.join(self.data_path, f'predict_data_smiles.txt'), 'r') as f:
                smiles_ = f.readlines()
                smiles_list = [s.strip() for s in smiles_]

            with open(osp.join(self.data_path, f"node_feats.pkl"), 'rb') as f:
                self.node_feats = pickle.load(f)

        else:
            print('preprocessing data ...')
            with open(self.data_path + 'predict_data.txt', 'r') as f:
                lines = f.readlines()

            smiless = [l.strip('\n') for l in lines]
            self.data_list, smiles_list, self.nnodes_list = [], [], []
            self.node_feats, self.edge_feats = [], []
            for smiles in smiless:

                mol = Chem.MolFromSmiles(smiles)
                cano_smiles = Chem.MolToSmiles(mol)
                data, node_feats = smiles_to_graph(cano_smiles, self.vocab)

                if torch.cuda.is_available:
                    data = data.to('cuda:0')  # move to GPU
                else:
                    data = data.to('cpu')  # move to CPU

                self.data_list.append(data)
                self.node_feats.append(node_feats)

                smiles_list.append(cano_smiles)
                self.nnodes_list.append(data.number_of_nodes())
            save_graphs(osp.join(self.data_path, f"predict_data.bin"), self.data_list)

            with open(osp.join(self.data_path, f"node_feats.pkl"), 'wb') as f:
                pickle.dump(self.node_feats, f)

            with open(osp.join(self.data_path, f"predict_data_smiles.txt"), 'w') as f:
                for smiles in smiles_list:
                    f.write(smiles + '\n')
            np.save(self.data_path + 'n_nodes.npy', self.nnodes_list)
        self.smiles_list = np.array(smiles_list)

        self.kpgt_feats = np.load(osp.join(self.data_path, f'kpgt_predict.npz'))
        self.kpgt_feats = self.kpgt_feats['fps_list'].tolist()

    def __len__(self):
        """Length of the dataset

        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.smiles_list)

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for tasks
        """
        return self.smiles_list[item], self.data_list[item], self.node_feats[item], self.kpgt_feats[item]