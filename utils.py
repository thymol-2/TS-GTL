import os
import numpy as np
import torch
import random
import dgl


def set_random_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def makedir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path, exist_ok=True)

def load_model(device=torch.device('cuda:0'),
               source_model_path=None):

    args = get_args()

    from model.PGnT import PGnT
    model = PGnT(is_classif=1, in_feats=138, args=args)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if source_model_path is not None:
        print(f"loading pretrained model: {source_model_path}")
        checkpoint = torch.load(source_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    return model



import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GAT model arguments")

    parser.add_argument('--hidden_dim', type=int, default=[1024, 1024, 1024], help='Dimension of hidden layers')   # 75维是[512,512,512]，138维参数改为[1024,1024,1024]
    parser.add_argument('--num_heads', type=int, default=[8,8,8], help='Number of attention heads')   # 最初是[8,8,8]
    parser.add_argument('--drops', type=float, default=[0.2, 0.2, 0.2], help='Dropout rate for features and attention')  # [0.2， 0.2， 0.2]
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for FPN')
    parser.add_argument('--fp_2_dim', type=int, default=1024, help='Dimension of the second fingerprint layer')  # 75维是512，138维参数改为1024
    parser.add_argument('--gat_scale', type=float, default=0.5, help='The ratio of gnn in model.')
    parser.add_argument('--out_feats', type=float, default=512, help='Output feature dimension.')   # 75维是512，138维大小参数都是512

    args = parser.parse_args()
    return args

