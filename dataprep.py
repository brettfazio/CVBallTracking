import numpy as np
import pandas as pd
import gc

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import os.path as osp
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as D

class Dataset (D.Dataset):
    def __init__(self, root):
        self.filenames = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.CenterCrop(15),
            transforms.ToTensor(),
            ])
        self.toTensor = transforms.ToTensor()
        filenames = glob.glob(osp.join(root, "*.jpg"))

        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        labels = np.genfromtxt('sample_data/labels.csv', delimiter=',')

        if labels[index, 1] == 1:
            label = np.array([1.0])
        else:
            label = np.array([0.0])

        image = self.transform(image)
        return image, label
    
    def __len__(self):

        return self.len

