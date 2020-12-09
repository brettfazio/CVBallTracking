"""

The following is using the model and some code basis from https://github.com/eriklindernoren/PyTorch-YOLOv3

It is a modification of his code to read in and use the pre-trained YOLOv3 model, modified to do video tracking.

"""


from yolo.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *

import os
import sys
import time
import datetime
import argparse
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import numpy as np

import cv2 as cv2

"""
detect(image) returns [bounding boxes]

Detect takes in an image and returns the bounding boxes of all balls in the image

"""

def detect(image):
    model_def = 'yolo/config/yolov3.cfg'
    weights_path = 'yolo/weights/yolov3.weights'
    class_path = 'yolo/data/coco.names'
    conf_threshold = 0.8
    nms_threshold = 0.4
    batch_size = 1
    n_cpu = 0
    image_size = 416 #320 #416    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(model_def, img_size=image_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    # Write the image to a temporary file to put into the dataloader
    path = Path(__file__).parent.absolute()
    path = os.path.join(path, 'temp')
    cv2.imwrite(os.path.join(path, 'in.jpg'), image)

    image_folder = 'temp'

    # Load the folder containing our frame
    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    # Array of bounding boxes of balls to return
    boxes = []

    images = []
    iter_detections =[]

    # Go through dataloader
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))

        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_threshold, nms_threshold)

        images.extend(img_paths)
        iter_detections.extend(detections)


    # Go through the detections
    for imgi, (path, detections) in enumerate(zip(images,iter_detections)):
        if not (detections is None) and not len(detections.size()) == 0:
            detections = rescale_boxes(detections, image_size, image.shape[:2])
            # Iterate over the bounding boxes
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # Check if this bbox is a sports ball, if so append to our list
                if classes[int(cls_pred)] == 'sports ball':
                    boxes.append([x1, y1, x2-x1, y2-y1])
              
    # Return a list of all detected sports ball bounding boxes
    return boxes
