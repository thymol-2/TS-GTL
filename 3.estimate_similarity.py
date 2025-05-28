import torch
from motse import MoTSE

from data.load_data import load_probe_data
from utils import set_random_seed

set_random_seed(42)


model_type = 'PGnT'
probe_data_path = './dataset/probe_data/zinc500/probe_data/'
device = torch.device("cuda:0")


motse = MoTSE(device)

probe_data = load_probe_data(probe_data_path)


source_tasks = ['logD']
target_tasks = ['HOB']
filename = 'logD'
source_model_paths = []
for task in source_tasks:
    source_model_paths.append(f"./saved_models/PGnT/logD/{task}.pth")
target_model_paths = []
for task in target_tasks:
    target_model_paths.append(f"./saved_models/PGnT/logS/{task}.pth")


motse.cal_sim(source_tasks,target_tasks,
              source_model_paths,target_model_paths,probe_data,
              filename
             )