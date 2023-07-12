import torch
import torchvision

def IoU(box1, box2):
    """
    Calculates the IoU of 2 boxes.
    boxes are of the format [x,y,h,w] where they are all pixels, not relative to box cells or a percentage of the image.
    """
    # To calculate the union, we first want to see the box that represents the union of the 2 boxes
    # This calculates the right most left edge of the 2 boxes
    innerX1 = max(box1[0], box2[0])
    innerY1 = max(box1[1], box2[1]) 

    box1x2 = box1[0] + box1[3]
    box2x2 = box2[0] + box2[3]
    innerX2 = min(box1x2, box2x2)  # this calculates the left most right edge of the 2 boxes
    if innerX1 >= innerX2: # if the right most left edge is larger than the left most right edge, then they do not intersect at all, we can short circuit and return a IoU of 0 early
        return 0

    box1y2 = box1[1] + box1[2]
    box2y2 = box2[1] + box2[2]
    innerY2 = min(box1y2, box2y2)
    if innerY1 >= innerY2:
        return 0

    innerWidth = innerX2 - innerX1
    innerHeight = innerY2 - innerY1
    intersection = innerWidth * innerHeight
    if intersection == 0:
        return 0

    # next we calculate the total area of the 2 squares. The intersections is total area - union
    box1Area = box1[2] * box1[3]
    box2Area = box2[2] * box2[3]
    totalArea = box1Area + box2Area
    union = totalArea - intersection
    IoU = intersection / union
    return IoU