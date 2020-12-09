import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv2

from detect import detect
from utility import compute_iou

"""
The following will run YOLO on every frame of the video to use it as a source 
of truth to perform IOU. 

This will be useful in the case where CRST is used to track the ball, to see if there
is any divergence. Possibly if a ball goes through a net or behind a hand, will CRST correct?
"""
def yolo_based_eval(video_file, mapped_predictions):
    # Read video
    video = cv2.VideoCapture(video_file)

    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    # Go through frames of the video
    frame_index = -1

    ious = []
    iou_counted = 0

    # Get video dimensions
    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4)) 
   
    size = (frame_width, frame_height) 

    # Initialize video output
    result = cv2.VideoWriter(f"{video_file}-eval-out.avi",  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         30, size) 

    while video.isOpened():
        frame_index += 1
        ok, frame = video.read()
        
        if not ok:
            break

        # Make sure that we have a prediction for this frame,
        # if we don't report this as a missed frame (assuming YOLO finds something)
        # If we do have a prediction for this frame, just proceed as normal

        # Get the YOLO set

        new_bboxes = detect(frame)
        
        if len(new_bboxes) == 0:
            # no detected balls to compare against. Cannot make any assumptions here.
            predicted_bbox = mapped_predictions[frame_index]
            if predicted_bbox:
                p1 = (int(predicted_bbox[0]), int(predicted_bbox[1]))
                p2 = (int(predicted_bbox[0] + predicted_bbox[2]), int(predicted_bbox[1] + predicted_bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                cv2.putText(frame, f"Predicted Box", (0,frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,0,0),2)
                result.write(frame)
            continue

        highest = 0
        highest_bbox = None
        # Find the BBox with the highest IOU score, that is the one we wish to compare against
        for bbox in new_bboxes:
            predicted_bbox = mapped_predictions[frame_index]

            iou_score = compute_iou(predicted_bbox, bbox)
            if iou_score > highest:
                highest = iou_score
                highest_bbox = bbox
            highest = max(highest, iou_score)

        # If no boxes overlap, report that        
        if highest == 0:
            continue

        # Otherwise use the highest overlap box
        # Add to the total
        ious.append(highest)
        iou_counted += 1

        p1 = (int(predicted_bbox[0]), int(predicted_bbox[1]))
        p2 = (int(predicted_bbox[0] + predicted_bbox[2]), int(predicted_bbox[1] + predicted_bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        p1 = (int(highest_bbox[0]), int(highest_bbox[1]))
        p2 = (int(highest_bbox[0] + highest_bbox[2]), int(highest_bbox[1] + highest_bbox[3]))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

        cv2.putText(frame, f"Predicted Box", (0,frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,0,0),2)
        cv2.putText(frame, f"Truth Box", (0,frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,0),2)
        result.write(frame)

    average_iou = sum(ious) / float(iou_counted)
    video.release() 
    result.release()
    plt.plot(ious)
    plt.show()

    # Return average iou
    return average_iou


"""
This evaluation method will also use IOU, but will be useful if either CRST or YOLO was used on every frame.

The below needs a labeled datasete to compare against the predictions.
"""
def bounded_eval(mapped_truth, mapped_predictions):
    return

def eval_precision(mapped_truth, mapped_predictions):
    index = 0
    positives = 0.0
    true_positives = 0.0

    for index in mapped_truth:
        if index in mapped_predictions.keys():
            positives += 1
            if compute_iou(mapped_truth[index], mapped_predictions[index]) > 0.5:
                true_positives += 1
                
    return true_positives / positives

def eval_recall(mapped_truth, mapped_predictions):
    index = 0
    false_negatives = 0.0
    true_positives = 0.0

    for index in mapped_truth:
        if index not in mapped_predictions.keys():
            false_negatives += 1
        elif compute_iou(mapped_truth[index], mapped_predictions[index]) > 0.5:
                true_positives += 1

    return true_positives / (true_positives + false_negatives)