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
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                self.optimizer.zero_grad()        
                y_pred = self.model(X_batch)

                loss = self.criterion(y_pred, y_batch.squeeze())
                acc = self.metric(y_pred, y_batch.squeeze())

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                print(self._progress(e, batch_idx, len(self.train_loader),loss), end='\r')
            ## Validation
            val_loss, val_acc = self._valid(e, device)
            # wandb log

    def _valid(self, epoch, device):
        self.model.eval()
        with torch.no_grad(): 
            val_loss = 0
            val_acc = 0
            for x_val, y_val in self.valid_loader:  
                x_val = x_val.to(device)  
                y_val = y_val.to(device)   

                yhat = self.model(x_val)  
                val_loss += self.criterion(yhat, y_val.squeeze()).item()
                acc = self.metric(yhat, y_val.squeeze())
                val_acc += acc.item()
        print(f"\n# Valid [Epoch {epoch}] Loss: {val_loss/len(self.train_loader):.04} | Acc: {val_acc/len(self.train_loader):.04}")
        return val_loss, val_acc


    def _progress(self, epoch, batch_idx, iter_len, loss):
        """
        args
            epoch: current epoch
            batch_idx: batch index
            loss: current epoch's loss
        """
        base = '[Epoch {}/{}] ({}/{} iters) loss: {:.04}'
        return base.format(epoch, self.epochs, batch_idx, iter_len, loss)