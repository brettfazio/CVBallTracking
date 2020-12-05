import numpy as np

from detect import detect

"""
The following will run YOLO on every frame of the video to use it as a source 
of truth to perform IOU. 

This will be useful in the case where CRST is used to track the ball, to see if there
is any diverage. Possibly if a ball goes through a net or behind a hand, will CRST correct?
"""
def yolo_based_eval(video_file, mapped_predictions):
    # Read video
    video = cv2.VideoCapture(video_file)

    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    # Go through frames of the video
    frame_index = 0

    while video.isOpened():
        ok, frame = video.read()

        if not ok:
            break

        # Make sure that we have a prediction for this frame,
        # if we don't report this as a missed frame (assuming YOLO finds something)
        # If we do have a prediction for this frame, just proceed as normal

        # Get the YOLO sot

        new_bboxes = detect(frame)
        
        # Find the BBox with the highest IOU score, that is the one we wish to compare against

        # If no boxes overlap, report that

        # Otherwise use the highest overlap box
        
        frame_index += 1

    # Return evaluation data
    return


"""
This evaluation method will also use IOU, but will be useful if either CRST or YOLO was used on every frame.

The below needs a labeled datasete to compare against the predictions.
"""
def bounded_eval(mapped_truth, mapped_predictions):
    return

