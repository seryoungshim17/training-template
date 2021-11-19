from easydict import EasyDict
import json
from models import loss
from models import metrics
from models import model
from models import optimizer
from custom_dataset import transformer
import wandb

def json_config(file_path):
    cfg = EasyDict()
    with open(file_path, 'r') as f:
        cfg.update(json.load(f))

    cfg.model = getattr(model, cfg.model)(cfg.num_classes)
    cfg.loss = getattr(loss, cfg.hyperparam.loss)
    cfg.metric = getattr(metrics, cfg.metric)
    
    cfg.optim = getattr(optimizer, cfg.hyperparam.optimizer.name)(cfg.model, cfg.hyperparam.lr, cfg.hyperparam.optimizer.weight_decay)
    cfg.transform = transformer.config_to_transform(cfg.transform)
    cfg.tta = transformer.config_to_transform(cfg.tta)
    
    return cfg

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