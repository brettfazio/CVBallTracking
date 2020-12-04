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
 
    # Get video dimensions
    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4)) 
   
    size = (frame_width, frame_height) 

    # Initialize video output
    result = cv2.VideoWriter(f"{file}-{tracker_type}-out.avi",  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         30, size) 
    
    # Initialize bbox output
    ret = {}

    # Hyper parameter - give extra space to the bounding box
    # this helps when the ball bounces off screen, goes through a net, or is behind a player's hand.
    extra_size = 8

    # Reshape bbox input
    bbox = (bbox[0]-extra_size, bbox[1]-extra_size, bbox[2]+extra_size*2, bbox[3]+extra_size*2)

    # Go through each frame
    current_frame = 0
    while True:
        
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break
        
        if current_frame < start:
            result.write(frame)
            current_frame += 1
            continue

        if current_frame == start:
            # Initialize tracker with starting frame and bounding box
            ok = tracker.init(frame, bbox)

        if not ok:
            print ('failed to init tracker')
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
        current_frame += 1

    # When everything done, release  
    # the video capture and video  
    # write objects 
    video.release() 
    result.release() 

    # Closes all the frames 
    cv2.destroyAllWindows()

    return ret

"""

Given a video file, as well as the frame from which to start and an initial bounding box,
this will use overlap based tracking to track the ball throughout the rest of the video.

Since this will be running localization on every frame it will not be as performant as the above
opencv based tracker.

"""
def overlap_track(file, start, bbox):
    return
