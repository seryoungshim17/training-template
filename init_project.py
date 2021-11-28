from easydict import EasyDict
import json
from src import loss
from src import metrics
from src import models
from src import optimizer
from custom_dataset import transformer
import torch
import numpy as np
import random
import wandb
from importlib import import_module

def json_config(file_path):
    cfg = EasyDict()
    with open(file_path, 'r') as f:
        cfg.update(json.load(f))

    # cfg.model = getattr(models, cfg.model)(cfg.num_classes)
    cfg.model = getattr(import_module("src.models."+cfg.model.split('.')[0]), cfg.model.split('.')[1])(cfg.num_classes)
    cfg.loss = getattr(loss, cfg.hyperparam.loss)
    cfg.metric = getattr(metrics, cfg.metric)
    
    cfg.optim = getattr(optimizer, cfg.hyperparam.optimizer.name)(cfg.model, cfg.hyperparam.lr, cfg.hyperparam.optimizer.weight_decay)
    cfg.transform = transformer.config_to_transform(cfg.transform)
    cfg.tta = transformer.config_to_transform(cfg.tta)
    
    return cfg
def init_seed(random_seed = 2021):
    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def wandb_init(cfg):
    if 'id' in cfg.log.wandb:
        id = cfg.log.wandb.id
    else:
        id = wandb.util.generate_id()
        cfg.id = id
    wandb.init(
        project=cfg.log.wandb.project,
        id = id,
        resume="allow",
        config=cfg
    )
    wandb.run.name = cfg.log.wandb.run_name
    wandb.run.save()