import cv2 as cv2

"""

Function to be used in multiple parts of our project. It computes the IOU score of two provided bounding boxes.

"""
def compute_iou(bbox_a, bbox_b):
    # find coordinates of intersection

    tl_x = min(bbox_a[0], bbox_b[0])
    tl_y = min(bbox_a[1], bbox_b[1])

    # since we input as x,y,w,h we need to compute the other coordinate
    br_x = max(bbox_a[0] + bbox_a[2], bbox_b[0] + bbox_b[2])
    br_y = max(bbox_a[1] + bbox_a[3], bbox_b[1] + bbox_b[3])

    # Take the max in case there is no overlap at all (would be negative)

    # todo(): do i need to do br_x - tl_x OR br_x - tl_x + 1
    inter_w = max(0, br_x - tl_x)

    inter_h = max(0, br_y - tl_y)

    inter_area = inter_w * inter_h

    a_area = bbox_a[2] * bbox_a[3]
    b_area = bbox_b[2] * bbox_b[3]

    iou = inter_area / float(a_area + b_area - inter_area)

    return iou

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_tracker(tracker_type):
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

    return tracker