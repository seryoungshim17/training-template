from easydict import EasyDict
import json
from models import loss
from models import metrics
from models import model
from models import optimizer
# import wandb

def json_config(file_path):
    cfg = EasyDict()
    with open(file_path, 'r') as f:
        cfg.update(json.load(f))

    cfg.model = getattr(model, cfg.model)(cfg.num_class)
    cfg.loss = getattr(loss, cfg.hyperparam.loss)
    cfg.metric = getattr(metrics, cfg.metric)
    
    cfg.optim = getattr(optimizer, cfg.hyperparam.optimizer.name)(cfg.model, cfg.hyperparam.lr, cfg.hyperparam.optimizer.weight_decay)
    return cfg

def wandb_init():
    return