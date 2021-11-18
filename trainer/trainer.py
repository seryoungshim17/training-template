import os
from tqdm import tqdm
import torch

class Trainer():
    """
    Trainer class
    method
        __init__
        _train : train model
        _valid : validate model
    """
    def __init__(self, cfg, train_loader, valid_loader):
        self.model = cfg.model
        self.criterion = cfg.loss
        self.optimizer = cfg.optim
        self.metric = cfg.metric
        self.epochs = cfg.hyperparam.epoch

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def _train(self, num_classes, device, saved_folder, wandb=None):
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
        self.model = self.model.to(device)

        for e in range(1, self.epochs+1):
            epoch_loss = 0
            epoch_acc = 0
            self.model.train()
            for (X_batch, y_batch) in tqdm(self.train_loader, desc=f"Epoch {e}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                self.optimizer.zero_grad()        
                y_pred = self.model(X_batch)

                loss = self.criterion(y_pred, y_batch.squeeze())
                acc = self.metric(y_pred, y_batch.squeeze())

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc
            print(epoch_loss, epoch_acc)
            ## Validation

    def _valid(self):
        pass

    def _progress(self, batch_idx):
        """
        args
            batch_idx: batch index
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)