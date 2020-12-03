import cv2
import sys


"""
   Outputs video of tracked object given:
    file: name of the video file
    tracker_type: tracking method to be used
    bbox: an initial bbox [top left corner, height, width]
    start: frame to start tracking
   Returns map of frame to bbox tracked in that frame
"""

def opencv_track(file, tracker_type, start, bbox):
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
    print("Using {} tracking", tracker_type)

    # Read video
    video = cv2.VideoCapture(file)

    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()

    # Start at specified frame
    video.set(cv2.CV_CAP_PROP_POS_FRAMES, start)
    current_frame = start

    # Initialize tracker with starting frame and bounding box
    ok = tracker.init(frame, bbox)

    # Initialize video output
    result = cv2.VideoWriter(f"{video}-{tracker_type}-out.mp4",  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, size) 
    
    # Initialize bbox output
    ret = {}

    # Go through each frame
    while True:
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)

        # Update bbox output
        ret[current_frame] = bbox
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
 
        # Display result, write to vid
        result.write(frame)
        cv2.imshow(f"{tracker_type} Tracking", frame)

    # When everything done, release  
    # the video capture and video  
    # write objects 
    video.release() 
    result.release() 

    # Closes all the frames 
    cv2.destroyAllWindows()

    return ret 
