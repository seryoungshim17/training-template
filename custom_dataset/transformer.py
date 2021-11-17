import torchvision.transforms as transforms
import torch

def config_to_transform(transform):
    transform_list = []
    for T in transform:
        transform_list.append(getattr(transforms, T.name)(**T.args))
    
    transform_list.append(transforms.PILToTensor())
    transform_list.append(transforms.ConvertImageDtype(torch.float))
    return transforms.Compose(transform_list)