import torch
import numpy as np

import tqdm
from tqdm import tqdm

from meter import Meter
from earlystopping import EarlyStopping
from scheduler import PolynomialDecayLR

from torch.optim.lr_scheduler import LambdaLR

class Trainer(object):
    def __init__(self, device, task, data_args, model_path,
                 n_epochs, patience=50, inter_print=20):
        self.device = device
        self.task = task
        self.n_tasks = len(self.task)
        self.model_path = model_path
        self.metrics = data_args['metrics']
        self.norm = data_args['norm']
        if self.norm:
            self.data_mean = torch.tensor(data_args['mean']).to(self.device)
            self.data_std = torch.tensor(data_args['std']).to(self.device)
        self.loss_fn = data_args['loss_fn']
        self.n_epochs = n_epochs
        self.patience = patience
        self.inter_print = inter_print


    def _prepare_batch_data(self, batch_data):
        smiless, inputs, labels, masks, node_feats, kpgt_feats = batch_data
        labels = labels.to(self.device)
        masks = masks.to(self.device)
        kpgt_feats = kpgt_feats.to(self.device)

        return smiless, inputs, labels, masks, kpgt_feats


    def _train_epoch(self, model, train_loader, loss_fn, optimizer):
        model.train()
        loss_list = []
        for i, batch_data in enumerate(train_loader):
            smiless, inputs, labels, masks, kpgt_feats = self._prepare_batch_data(batch_data)

            _, predictions = model(inputs, kpgt_feats)
            if self.norm:
                labels = (labels - self.data_mean)/self.data_std
            loss = (loss_fn(predictions, labels)*(masks!=0).float()).mean()

            optimizer.zero_grad()
            optimizer.step()
            loss_list.append(loss.item())

        return np.mean(loss_list)


    def _eval(self, model, data_loader):
        model.eval()
        meter = Meter(self.task)
        for i, batch_data in enumerate(data_loader):
            smiless, inputs, labels, masks, kpgt_feats = self._prepare_batch_data(batch_data)

            _, predictions = model(inputs, kpgt_feats)
            if self.norm:
                predictions = predictions * self.data_std + self.data_mean
            meter.update(predictions, labels, masks)
        eval_results_dict = meter.compute_metric(self.metrics)
        return eval_results_dict

    def _train(self, model, train_loader, val_loader, loss_fn, optimizer, stopper):
        train_losses = []
        for epoch in tqdm(range(self.n_epochs)):
            loss = self._train_epoch(model, train_loader, loss_fn, optimizer)
            val_results_dict = self._eval(model, val_loader)
            early_stop = stopper.step(val_results_dict[self.metrics[0]]['mean'],
                                     model, epoch)
            if epoch % self.inter_print == 0:
                print(f"[{epoch}] training loss:{loss}")
                for metric in self.metrics:
                    print(f"val {metric}:{val_results_dict[metric]['mean']}")
            if early_stop:
                break
            train_losses.append(loss)
        return train_losses



    def fit(self, model, train_loader, val_loader, test_loader):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()),
                                     lr=1e-4, weight_decay=1e-5)

        stopper = EarlyStopping(self.model_path, self.task, patience=self.patience)

        train_losses = self._train(model, train_loader, val_loader,
                    self.loss_fn, optimizer, stopper)
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Convergence')
        plt.legend()
        plt.show()

        stopper.load_checkpoint(model)
        test_results_dict = self._eval(model, test_loader)
        for metric in self.metrics:
            print(f"test {metric}:{test_results_dict[metric]['mean']}")
        return model, test_results_dict




