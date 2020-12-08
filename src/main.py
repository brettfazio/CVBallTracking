"""

Example run:

python main.py --video ../sample_data/1LdhIsz6INQ.mp4

"""

import argparse
import cv2 as cv
import sys
import numpy as np

from tracking import opencv_track, overlap_track
from detect import detect
from evaluate import yolo_based_eval

def yolo_track(video_path):
    mapped_results = overlap_track(video_path)
    
    return mapped_results

def track(video_path):

    video = cv.VideoCapture(video_path)

    bounding = np.array([])
    index = 0

    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break

        bbox = detect(frame)

        # For now just use the first bounding box found
        if len(bbox) > 0:
            bounding = bbox[0]
            break

        index += 1
    video.release() 
    # Now that we have the bounding box of the ball we can run opencv_track
    mapped_results = opencv_track(video_path, 'CSRT', index, bounding)

   return mapped_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Specify video source
    parser.add_argument("--video", type=str, default="video.mp4", help="video to track")
    # Specify mode (either yolo or track). yolo runs localization on every frame
    # track just tracks it after it finds the initial box.
    parser.add_argument("--mode", type=str, default="track", help='yolo or track')
    # Specify evaluation method (or if eval should be performed at all)
    parser.add_argument("--eval", type=str, default='none', help='none or yolo or sot')
    opt = parser.parse_args()
    print(opt.video)


    video = cv.VideoCapture(opt.video)

    # Perform specified tracking/localization mode
    if opt.mode == 'track':
        mapped_results = track(opt.video)
    else:
        mapped_results = yolo_track(opt.video)

    # Evaluate using specified method
    if opt.mode == 'yolo':
        avg_iou = yolo_based_eval(opt.video, mapped_results)
        print('YOLO-Based Average IOU computed as: ', avg_iou)
