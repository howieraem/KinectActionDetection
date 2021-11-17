import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class RGBDataset(Dataset):
    def __init__(self, directory: str):
        print('Start initializing dataset.')
        directory.replace('\\', '/')
        if directory[-1] != '/':
            directory += '/'
        self.root_dir = directory

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
