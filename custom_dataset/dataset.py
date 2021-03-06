import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, train=True, mode=False):
        assert os.path.exists(img_dir)
        self.transform = transform
        self.target_transform = target_transform

        self.image_paths = []
        self.labels = []

        # save dataset image paths and labels
        for dirs in os.listdir(img_dir):
            assert len(dirs) > 0, "No directories"
            for dir_name in dirs:
                for file in os.listdir(os.path.join(img_dir, dir_name)):
                    self.image_paths.append(os.path.join(img_dir, dir_name, file))
                    self.labels.append(int(dir_name))
                    if mode:
                        break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]))

        return image, self.labels[idx]

if __name__ == '__main__':
    train_dataset = CustomImageDataset('../data/mnist_png/training')
    print(train_dataset[0])
