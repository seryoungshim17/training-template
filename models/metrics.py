import torchmetrics.functional as F
import torch

def accuracy(y_pred, target):
    y_pred = torch.argmax(y_pred, dim=1)
    return F.accuracy(y_pred, target)

def f1_score(y_pred, target, num_classes):
    y_pred = torch.argmax(y_pred, dim=1)
    return F.f1(y_pred, target, average='macro', num_classes=num_classes)
