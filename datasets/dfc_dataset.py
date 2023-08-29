import os
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DFCDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.data_path = 'dfc-data/'
        self.image_ids = os.listdir(f'{self.data_path}/view1/')

        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        view1 = Image.open(f'{self.data_path}/view1/{image_id}')
        mask = Image.open(f'{self.data_path}/maskgt9/{image_id}')

        if self.transform:
            view1 = self.transform(view1)
            mask = self.transform(mask)

        return view1, mask

