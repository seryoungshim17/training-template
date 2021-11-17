import torch
def adam(model, lr, weight_decay):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)