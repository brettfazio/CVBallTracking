import argparse
import cv2 as cv
import sys

from tracking import opencv_track
from detect import detect

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video.mp4", help="video to track")

    opt = parser.parse_args()
    print(opt.video)


    video = cv.VideoCapture(opt.video)
    ok, frame = video.read()

    if not ok:
        print ('Cannot read video file')
        sys.exit()

    bboxes = detect(frame)