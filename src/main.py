"""

Example run:

python main.py --video ../sample_data/1LdhIsz6INQ.mp4

"""

import argparse
import cv2 as cv
import sys
import numpy as np

from tracking import opencv_track
from detect import detect

def yolo_track(video_path):
    
    video = cv.VideoCapture(video_path)

    while video.isOpened():
        ok, frame = video.read()

        if not ok:
            break

        bbox = detect(frame)

    

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
            bounding = bbox
            break

        index += 1
    video.release() 
    # Now that we have the bounding box of the ball we can run opencv_track
    mapped_results = opencv_track(video_path, 'CSRT', index, bounding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Specify video source
    parser.add_argument("--video", type=str, default="video.mp4", help="video to track")
    # Specify mode (either yolo or track). yolo runs localization on every frame
    # track just tracks it after it finds the initial box.
    parser.add_argument("--mode", type=str, default="track", help='yolo or track')

    opt = parser.parse_args()
    print(opt.video)


    video = cv.VideoCapture(opt.video)

    if opt.mode == 'track':
        track(opt.video)
    else:
        yolo_track(opt.video)
