from init_project import json_config, wandb_init, init_seed
from custom_dataset.dataset import CustomImageDataset
from src.trainer import Trainer
import torch
from torch.utils.data import DataLoader
import argparse
import pathlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config file path')
    parser.add_argument(
        '--config', 
        type=pathlib.Path,
        default='./config/__base__.json'
    )
    args = parser.parse_args()

    cfg = json_config(file_path=args.config)
    init_seed(cfg.seed)
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

    ## WandB Init
    if 'wandb' in cfg.log:
        wandb_init(cfg)

    ## Start training
    T = Trainer(cfg, train_loader, valid_loader)
    T._train(cfg.num_classes, DEVICE, cfg.log.model_dir, 'wandb' in cfg.log)
