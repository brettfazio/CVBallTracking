import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models, utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.ndimage.filters import convolve
from queue import Queue
import cv2
import pandas as pd
from SoccerDataset import SoccerDataset
from ConvNet import ConvNet

def train(model, train_loader, optimizer, criterion):
    model.train()


    model = model.double()
        
    # Empty list to store losses 
    losses = []
    correct = 0
    tot = 0
    computed_loss = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        # Push data/label to correct device
        #print(data.shape)
        #data, target = data.to(device), target.to(device)
        
        data = batch_sample['image']
        target = batch_sample['landmarks']

        # input[10, 600, 800, 3] 
        data = data.view(-1, 3, 150, 200)

        data = data.double()
        target = target.double()

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        
        # ======================================================================
        # Count correct predictions overall 
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign counting function for correct predictions.
        computed_loss += loss.item()
        
    train_loss = computed_loss / len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}\n'.format(
        float(np.mean(losses))))
    return train_loss, train_acc

def prepare_data():
    dataset = SoccerDataset('../export_2595_bitbots-nagoya-sequences-jasper-euro-ball-1.csv')    
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [dataset.__len__()-250, 250])

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_data, test_data = prepare_data()

    train_loader = DataLoader(train_data, batch_size=10)

    test_loader = DataLoader(test_data, batch_size=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model, optimizer, and criterion
    model = ConvNet(1).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # run training for n_epochs specified
    epochs = 5

    for epoch in range(epochs):
        # train
        train(model, train_loader,optimizer, criterion)
