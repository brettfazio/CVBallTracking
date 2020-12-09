import cv2 as cv2

import pandas as pd
import os
import numpy as np

# for parsing a2d. not needed if just running on video input
import h5py

PATH_TO_A2D_CSV = '../a2d/videoset.csv'

"""

Function to be used in multiple parts of our project. It computes the IOU score of two provided bounding boxes.

"""
def compute_iou(bbox_a, bbox_b):
    # find coordinates of intersection

    tl_x = max(bbox_a[0], bbox_b[0])
    tl_y = max(bbox_a[1], bbox_b[1])

    # since we input as x,y,w,h we need to compute the other coordinate
    br_x = min(bbox_a[0] + bbox_a[2], bbox_b[0] + bbox_b[2])
    br_y = min(bbox_a[1] + bbox_a[3], bbox_b[1] + bbox_b[3])

    # Take the max in case there is no overlap at all (would be negative)

    # todo(): do i need to do br_x - tl_x OR br_x - tl_x + 1
    inter_w = max(0, br_x - tl_x)

    inter_h = max(0, br_y - tl_y)

    inter_area = inter_w * inter_h

    a_area = bbox_a[2] * bbox_a[3]
    b_area = bbox_b[2] * bbox_b[3]

    iou = inter_area / float(a_area + b_area - inter_area)

    return iou


"""
Function used to convert strings to bools for use in main argument flags
"""
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

"""
Helper method to allow the user to textually specify their desired tracker
"""
def get_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    return tracker

"""
Parses the a2d csv file to get all videos with localized balls
"""
def get_a2d_df():
    videoset_path = PATH_TO_A2D_CSV

    videoset = pd.read_csv(videoset_path)

    # Ball videos in the dataset will have ID of 34, 35, 36, 39
    # flying is 34, jumping is 35, rolling is 36, none is 39.

    # First get the names of all the ball videos.

    # Do this so we can more easily index the column
    videoset['Label_str'] = videoset['Label'].astype(str)

    # Only get ball videos
    videoset = videoset[videoset.Label_str.str.startswith('3')]

    # At this point all rows are referecning ball videos.
    # 253 are classified as training samples. 62 are classified as testing samples based on the provided usage.

    train_df = videoset[videoset['Usage'] == 0]
    test_df = videoset[videoset['Usage'] == 1]

    return videoset

"""
A2d provides labeled bounding boxes in compiled MatLab files, this method parses
said files and returns the bounding box contained in them
"""
def get_matlab_bboxes(path):
    # first parse out frame number
    base = os.path.basename(path)

    file_name = os.path.splitext(base)[0]

    frame_number = int(file_name)

    # Now find the ball bounding boxes
    bboxes = []

    f = h5py.File(path, 'a')

    itrs = f['class'].shape
    itrs = itrs[1]

    for i in np.arange(itrs):
        obj = f['class'][0][i]
        s = ''

        # only read 3 chars cause 'dog' is a class
        for k in np.arange(3):
            s += chr(f[obj][k][0])
        
        if s == 'bal':
            # Use bbox at this index
            # They are column based
            bbox = []
            for idx in np.arange(4):
                bbox.append(f['reBBox'][idx][i])
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            bboxes.append(bbox)

    return bboxes, frame_number

def reshape_to_rect(bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

    return p1, p2
