import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
import argparse 

from data.featurizer import smiles_to_graph_tune
from data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args

def preprocess_dataset(data_path, dataset, path_length=5, n_jobs=1):
    df = pd.read_csv(f"{data_path}/{dataset}.csv")
    cache_file_path = f"{data_path}/{dataset}_{path_length}.pkl"
    smiless = df['Smiles'].values.tolist()
    task_names = df.columns.drop(['Smiles']).tolist()
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                   smiless,
                   max_length=path_length,
                   n_virtual_nodes=2,
                   n_jobs=n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]

    save_graphs(cache_file_path, valid_graphs,
                labels={'HOB': labels})

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"{data_path}/{dataset}_rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{data_path}/{dataset}_molecular_descriptors.npz",md=arr[:,1:])

if __name__ == '__main__':

    dataset = 'probe_data'
    data_path = '../dataset/probe_data/zinc500/probe_data'
    preprocess_dataset(data_path, dataset)