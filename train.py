from init_project import json_config
from custom_dataset.dataset import CustomImageDataset
from trainer.trainer import Trainer
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    cfg = json_config(file_path='./config/__base__.json')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Prepare dataset
    train_dataset = CustomImageDataset('./data/mnist_png/training/', cfg.transform, cfg.tta, train=True, mode=True)
    valid_dataset = CustomImageDataset('./data/mnist_png/testing/', cfg.transform, cfg.tta, train=True, mode=True)
    # train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=cfg.hyperparam.batch, shuffle=False, 
                                num_workers=cfg.hyperparam.num_workers, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.hyperparam.batch, shuffle=False, 
                                num_workers=cfg.hyperparam.num_workers, drop_last=False)

    ## Start training
    T = Trainer(cfg, train_loader, valid_loader)
    T._train(cfg.num_classes, DEVICE, cfg.log.model_dir)