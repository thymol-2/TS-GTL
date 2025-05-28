import torch
import pandas as pd
from data.load_data import load_data

from utils import load_model, makedir
from trainer import Trainer

dataset = 'logD'
data_path = './dataset/logD/'
tasks = ['logD']
model_path = './saved_models/PGnT/logD/'


makedir(model_path)
device = torch.device("cuda:0")
results_dict = {'task':[]}


for task in tasks:
    print(task)
    train_loader, val_loader, test_loader, data_args = load_data(
        dataset=dataset,
        data_path=data_path,
        task=[task],
        device=device
    )
    model = load_model(device=device)
    trainer = Trainer(device=device,task=[task],
                      data_args=data_args,model_path=model_path, n_epochs=500)
    model, task_results_dict = trainer.fit(model, train_loader, val_loader, test_loader)
    results_dict['task'].append(task)
    for metric in data_args['metrics']:
        if metric not in list(results_dict.keys()):
            results_dict.update({metric:[]})
        results_dict[metric].append(task_results_dict[metric][task])


result_path = model_path.replace('saved_models','results')
makedir(result_path)
pd.DataFrame(results_dict).to_csv(result_path+'results.csv', float_format='%.4f',
                                  index=False)
print(f"Results have been saved to {result_path+'results.csv'}")

