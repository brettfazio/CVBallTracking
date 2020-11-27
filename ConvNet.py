import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        # Network architecture for the first part of question 1
        # Comprised of 3 convolutional layers, utilizng dropout, batch normalization, and relu
        self.conv_layer = nn.Sequential(

            nn.Conv2d(3, 8, kernel_size=9, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=9, stride=1),
            nn.MaxPool2d(kernel_size=9, stride=1),

            nn.Conv2d(8, 16, kernel_size=9, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=9, stride=1),
            nn.MaxPool2d(kernel_size=9, stride=1),

            nn.Conv2d(16, 32, kernel_size=9, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=9, stride=1),

            nn.Dropout2d(p=0.1),
        )


        # Fully connected layers
        # Adjust size of input for image size
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(418048, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4),
            nn.Dropout(p=0.1),
        )

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # 3 convolutional layers
    def model_1(self, X):
        # conv layers
        X = self.conv_layer(X)
        print(X.shape)
        # flatten input
        X = X.view(X.size(0), -1)
        
        # fully connected layers
        X = self.fc_layer(X)

        # No softmax because its already included in nn.CrossEntropyLoss()
        return X
