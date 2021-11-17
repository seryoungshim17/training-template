from easydict import EasyDict
import json
from models import loss
from models import metrics
from models import model

def JsonConfig(file_path):
    cfg = EasyDict()
    with open(file_path, 'r') as f:
        cfg.update(json.load(f))

    cfg.model = getattr(model, cfg.model)
    cfg.loss = getattr(loss, cfg.hyperparam.loss)
    cfg.metric = getattr(metrics, cfg.metric)

    
    return cfg