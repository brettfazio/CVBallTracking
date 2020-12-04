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
