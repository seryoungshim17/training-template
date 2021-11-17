from init_project import json_config
from custom_dataset.dataset import CustomImageDataset
import torch

if __name__ == '__main__':
    cfg = json_config(file_path='./config/__base__.json')
    print(cfg)

    # Prepare dataset
    datasets = CustomImageDataset('./data/mnist_png/training/', cfg.transform, cfg.tta)
    # train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])