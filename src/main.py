"""

Example run:

python main.py --video ../sample_data/1LdhIsz6INQ.mp4

"""

import argparse
import cv2 as cv
import sys

from tracking import opencv_track
from detect import detect

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
    ok, frame = video.read()

    if not ok:
        print ('Cannot read video file')
        sys.exit()

    
    bboxes = detect(frame)
