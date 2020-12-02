import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Network architecture for the first part of question 1
        # Comprised of 3 convolutional layers, utilizng dropout, batch normalization, and relu
        self.conv_layer = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

        )


        # Fully connected layers
        # Adjust size of input for image size
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(64 * 9, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
            nn.Dropout(p=0.1),
        )
        
    # 3 convolutional layers
    def forward(self, X):
        # conv layers
        X = self.conv_layer(X)

        # flatten input
        X = X.view(X.size(0), -1)
        
        # fully connected layers
        X = self.fc_layer(X)

        return X
net = ConvNet()

    
    
