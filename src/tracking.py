import cv2
import sys

from detect import detect
from utility import get_tracker

"""
   Outputs video of tracked object given:
    file: name of the video file
    tracker_type: tracking method to be used
    bbox: an initial bbox [top left corner, height, width]
    start: frame to start tracking
   Returns map of frame to bbox tracked in that frame
"""

def opencv_track(file, tracker_type, start, bbox, fast, live):
    tracker = get_tracker(tracker_type)
    print(f"Using {tracker_type} tracking")

    # Read video
    video = cv2.VideoCapture(file)

    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
    
    # live tracking only works on fast mode
    if live and not fast:
        live = False
        print("Live tracking only works on fast mode\n")
        
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
    # lebron 8
    extra_size = 8

    # Reshape bbox input
    bbox = (bbox[0]-extra_size, bbox[1]-extra_size, bbox[2]+extra_size*2, bbox[3]+extra_size*2)
    initial_bbox = bbox

    # Go through each frame
    current_frame = 0
    backwards_frames = list()
    forwards_frames = list()
    while True:
        
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break
        
        if current_frame < start:
            if not fast:
                backwards_frames.insert(0, frame)
            current_frame += 1
            continue

        if current_frame == start:
            # Initialize tracker with starting frame and bounding box
            ok = tracker.init(frame, bbox)

        if not ok:
            print ('failed to init tracker')
            break
         
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
        forwards_frames.append(frame)
        current_frame += 1
        if live:
            cv2.imshow(f"{tracker_type} Tracking", frame)
        
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    if backwards_frames:
        backwards = backwards_track(backwards_frames, tracker_type, initial_bbox)
        for frame in backwards[0]:
            result.write(frame)
    

    for frame in forwards_frames:
        result.write(frame)

    # When everything done, release  
    # the video capture and video  
    # write objects 
    video.release() 
    result.release() 

    # Closes all the frames 
    cv2.destroyAllWindows()

    return ret

def backwards_track(frames, tracker_type, bbox):
    tracker = get_tracker(tracker_type)

    bbox_results = {}
    frame_results = list()
    ok = tracker.init(frames[0], bbox)
    if not ok:
        print ('failed to init tracker')
        return 
    current_frame = 0
    for frame in frames:
 
        # Update tracker
        ok, bbox = tracker.update(frame)

        # Update bbox output
        bbox_results[current_frame] = bbox
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
 
        # Display result, write to vid
        frame_results.insert(0, frame)
        current_frame += 1
    return frame_results, bbox_results

"""

Given a video file, as well as the frame from which to start and an initial bounding box,
this will use overlap based tracking to track the ball throughout the rest of the video.

Since this will be running localization on every frame it will not be as performant as the above
opencv based tracker.

"""
def overlap_track(file):

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
    result = cv2.VideoWriter(f"{file}-overlap-out.avi",  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         30, size)

     # Initialize bbox output
    ret = {}

    # Go through each frame
    current_frame = 0
    while True:
        
        # Read a new frame
        ok, frame = video.read()
        print(current_frame)
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        if ok:
            bbox = detect(frame)
        else:
            break
        # Update bbox output
        ret[current_frame] = bbox
 
        # Draw bounding box
        if bbox:
            bbox = bbox[0]
            # Tracking success
            bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
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
