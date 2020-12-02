import sys
import numpy as np
import cv2 as cv
import os
from ConvNet import ConvNet 
import torch

size = 15
dim = 3

def check_pic(pic):
  img = cv.resize(pic, (size,size))
  img = torch.tensor(img)
  img = img.reshape(1, 3, size, size)
  img = img.float()
  
  return model(img)

model = ConvNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()


