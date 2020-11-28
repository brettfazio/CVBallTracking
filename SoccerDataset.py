import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SoccerDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = '../imageset_157/' + self.df.iloc[idx]['filename']

        image = io.imread(img_name)

        image = image.astype('double')

        image = transform.resize(image, (image.shape[0] // 4, image.shape[1] // 4),
                       anti_aliasing=True)

        # Last 4 elements are bounding box
        label_elements = self.df.iloc[idx][-4:]

        label = np.asarray(label_elements)

        label = label.astype('double').reshape(4)

        item = {'image': image, 'landmarks': label}

        if self.transform:
            item = self.transform(item)

        return item
