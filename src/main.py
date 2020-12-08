"""

Example run:

python main.py --video ../sample_data/1LdhIsz6INQ.mp4

"""

import argparse
import cv2 as cv
import sys
import numpy as np
import pandas as pd
import os
import glob

from tracking import opencv_track, overlap_track
from detect import detect
from evaluate import yolo_based_eval
from utility import str2bool, get_a2d_df, get_matlab_bboxes, compute_iou

def yolo_track(video_path):
    mapped_results = overlap_track(video_path)
    
    return mapped_results

def track(video_path, fast, live):

    video = cv.VideoCapture(video_path)

    bounding = np.array([])
    found = False
    index = 0

    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        bbox = detect(frame)

        # For now just use the first bounding box found
        if len(bbox) > 0:
            bounding = bbox[0]
            found = True
            break

        index += 1
        if index % 10 == 0:
            print(f"Running detect on frame {index}\n")

    video.release() 

    if found:
        # Now that we have the bounding box of the ball we can run opencv_track
        print(f"Detected ball on frame {index}\n")
        mapped_results = opencv_track(video_path, 'CSRT', index, bounding, fast, live)
        return mapped_results
    else:
        return {}

def run_a2d(amt, verbose):
    df = get_a2d_df()

    cnt = 0

    # Iterate over all videos in a2d
    for index, row in df.iterrows():
        if cnt > amt:
            break
        cnt += 1
        # This is the video ID
        vid = row['VID']

        # Path of the video itself
        path = '../a2d/Release/clips320H/' + vid + '.mp4'

        print(path)

        # Run tracking on that video to get the mapped predictions
        mapped_results = track(path, False, False)

        # Now that we have the mapped prediction, we will have to do some semi-complex parsing
        # of the matlab files to get the bounding boxes. A2D only gives the BBoxes in matlab.
        mats_folder = '../a2d/Release/Annotations/mat/' + vid + '/'

        # Paths of each of the matrices
        mats_paths = glob.glob(mats_folder + '*.mat')

        ious = []

        for path in mats_paths:
            bboxes, frame_number = get_matlab_bboxes(path)
            
            highest_iou = 0


            if verbose:
                print('On frame ', frame_number, ' mapped = ', mapped_results[frame_number])
                print('Bboxes on frame ', frame_number, ': ', bboxes)

            # Might be multiple balls, just use the one with the highest iou (if one exists)
            for bbox in bboxes:
                
                iou = compute_iou(bbox, mapped_results[frame_number])
                highest_iou = max(iou, highest_iou)
            
            ious.append(highest_iou)

        avg_iou = sum(ious) / float(len(ious))
        
        print('Avg IOU = ' + str(avg_iou))

    
    print('Completed a2d run')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Specify video source
    parser.add_argument("--video", type=str, default="video.mp4", help="video to track")
    # Specify mode (either yolo or track). yolo runs localization on every frame
    # track just tracks it after it finds the initial box.
    parser.add_argument("--mode", type=str, default="track", help='yolo or track')
    # Specify evaluation method (or if eval should be performed at all)
    parser.add_argument("--eval", type=str, default='none', help='none or yolo or sot')

    parser.add_argument("--fast", type=str2bool, nargs='?', const=True, default=False, help="Forwards pass only")

    parser.add_argument("--live", type=str2bool, nargs='?', const=True, default=False, help="Show results live")

    # Should it run on a2d dataset
    parser.add_argument('--a2d', type=str2bool, nargs='?', const=True, default=False, help='Run on the A2D dataset')
    # A2D amount
    parser.add_argument('--a2d_amt', type=int, default=10000, help='How many a2d samples to run on')

    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help='Enables extra logging')

    opt = parser.parse_args()
    print(opt.video)

    # Run and evaluate on the a2d dataset
    if opt.a2d:
        run_a2d(opt.a2d_amt, opt.verbose)
        sys.exit()
    
    video = cv.VideoCapture(opt.video)

    # Perform specified tracking/localization mode
    if opt.mode == 'track':
        mapped_results = track(opt.video, opt.fast, opt.live)
    else:
        mapped_results = yolo_track(opt.video)

    # Evaluate using specified method
    if opt.eval == 'yolo':
        avg_iou = yolo_based_eval(opt.video, mapped_results)
        print('YOLO-Based Average IOU computed as: ', avg_iou)
