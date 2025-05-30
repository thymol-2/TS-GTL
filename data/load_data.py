import dgl

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np


def load_data(dataset, data_path, task, device, load=True):
    data_args = dict()
    if dataset == 'hob':
        from data.hob import HobDataset
        train_set = HobDataset(data_path, 'train', task, load=load)
        test_set = HobDataset(data_path, 'test', task, load=load)
        val_set = HobDataset(data_path,'val', task, load=load)

    elif dataset == 'logP':
        from data.logP import logPDataset
        train_set = logPDataset(data_path, 'train', task, load=load)
        test_set = logPDataset(data_path, 'test', task, load=load)
        val_set = logPDataset(data_path,'val', task, load=load)

    elif dataset == 'logS':
        from data.logS import logSDataset
        train_set = logSDataset(data_path, 'train', task, load=load)
        test_set = logSDataset(data_path, 'test', task, load=load)
        val_set = logSDataset(data_path,'val', task, load=load)

    elif dataset == 'logD':
        from data.logD import logDDataset
        train_set = logDDataset(data_path, 'train', task, load=load)
        test_set = logDDataset(data_path, 'test', task, load=load)
        val_set = logDDataset(data_path,'val', task, load=load)


    data_args['metrics'] = train_set.metrics

    if train_set.task_type == 'classification':
        task_pos_weights = train_set.task_pos_weights
        data_args['norm'] = None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(device),
                                       reduction='none')
    else:
        data_args['task_pos_weights'] = None
        data_args['norm'] = train_set.norm
        if data_args['norm']:
            data_args['mean'] = train_set.mean
            data_args['std'] = train_set.std
        loss_fn = nn.MSELoss(reduction='none')

    data_args['loss_fn'] = loss_fn
    batch_size = train_set.batch_size

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, collate_fn=collate_molgraphs,
                              shuffle=True,
                              drop_last=True, num_workers=0)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, collate_fn=collate_molgraphs,
                              shuffle=True,
                             drop_last=True, num_workers=0)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, collate_fn=collate_molgraphs,
                            shuffle=False,
                            num_workers=0)

    return train_loader, val_loader, test_loader, data_args


def collate_molgraphs(data):

    assert len(data[0]) in [4, 5, 6], \
        'Expect the tuple to be of length 5 or 6, got {:d}'.format(len(data[0]))
    if len(data[0]) == 5:
        smiles, graphs, labels, node_feats, kpgt_feats = map(list, zip(*data))
        masks = None
    elif len(data[0]) == 6:
        smiles, graphs, labels, masks, node_feats, kpgt_feats = map(list, zip(*data))
    elif len(data[0]) == 4:
        smiles, graphs, node_feats, kpgt_feats = map(list, zip(*data))

        bg = dgl.batch([dgl.add_self_loop(g) for g in graphs])
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        node_feats_tensor = torch.cat(node_feats, dim=0).to('cuda')
        bg.ndata['node_pka'] = node_feats_tensor
        return smiles, bg, node_feats, kpgt_feats

    bg = dgl.batch([dgl.add_self_loop(g) for g in graphs])
    bg = bg.to('cuda')

    node_feats_tensor = torch.cat(node_feats, dim=0).to('cuda')
    bg.ndata['node_pka'] = node_feats_tensor


    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    kpgt_feats = torch.Tensor(kpgt_feats)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks, node_feats, kpgt_feats


def load_predict_data(data_path, batch_size=256, load=True):
    from data.predict_data import PredictDataset

    predict_data = PredictDataset(data_path, load=load)

    data_args = dict()
    data_args['metrics'] = predict_data.metrics

    predict_data = DataLoader(dataset=predict_data, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_molgraphs,
                            drop_last=False, num_workers=0)

    return predict_data, data_args

def load_probe_data(data_path, load=True):

    from data.probe_data import ProbeDataset

    probe_data = ProbeDataset(data_path,load=load)
    probe_data = DataLoader(dataset=probe_data, batch_size=probe_data.batch_size,
                              shuffle=False, collate_fn=collate_molgraphs,
                              drop_last=False, num_workers=0)
    nnode_list = np.load(data_path+'n_nodes.npy')
    probe_data = next(iter(probe_data))

    probe_data = {
        'data':probe_data,
        'name':data_path.split('/')[-2],
        'nnode':nnode_list,
    }
    return probe_data


