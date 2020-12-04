import numpy as np

"""
The following will run YOLO on every frame of the video to use it as a source 
of truth to perform IOU. 

This will be useful in the case where CRST is used to track the ball, to see if there
is any diverage. Possibly if a ball goes through a net or behind a hand, will CRST correct?
"""
def yolo_based_eval(video, mapped_predictions):
    return


"""
This evaluation method will also use IOU, but will be useful if either CRST or YOLO was used on every frame.

The below needs a labeled datasete to compare against the predictions.
"""
def bounded_eval(mapped_truth, mapped_predictions):
    return

