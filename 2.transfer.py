import torch
import pandas as pd
from utils import load_model, makedir, set_random_seed
from data.load_data import load_data
from trainer import Trainer

set_random_seed(42)

dataset = 'hob'
source_tasks = ['logD']
target_tasks = ['HOB']
data_path = './dataset/hob'
model_type = 'PGnT'
source_model_path = f"./saved_models/PGnT/logD/"
model_path = f"./saved_models/PGnT/logD-HOB/"
makedir(model_path)
device = torch.device('cuda:0')
results_dict = dict()


for target_task in target_tasks:
    results_dict[target_task] = {'source task':[]}
    for source_task in source_tasks:
        print((f"{source_task}->{target_task}"))

        train_loader, val_loader, test_loader, data_args = load_data(
            dataset=dataset,
            data_path=data_path,
            task=[target_task],
            device=device
        )

        model = load_model(device=device,
            source_model_path=source_model_path+f"{source_task}.pth")


        trainer = Trainer(
            device=device, task=[target_task],
            data_args=data_args, model_path=model_path, n_epochs=500
        )
        _, task_results_dict = trainer.fit(model, train_loader, val_loader, test_loader)

        results_dict[target_task]['source task'].append(source_task)
        for metric in data_args['metrics']:
            if metric not in list(results_dict[target_task].keys()):
                results_dict[target_task].update({metric: []})
            results_dict[target_task][metric].append(task_results_dict[metric][target_task])
    result_path = model_path.replace('saved_models', 'results')
    makedir(result_path)
    pd.DataFrame(results_dict[target_task]).to_csv(
        result_path + f'{target_task}.csv', float_format='%.4f', index=False)
    print(f"Results have been saved to {result_path + target_task + '.csv'}")


