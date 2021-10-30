import numpy as np

from typing import List


def to_xy_min_max(bboxes: np.ndarray) -> List[int]:
    """
    Convert one bounding box whose form is: [x1, y1, x2, y2, x3, y3, x4, y4] into a box of form (xmin, ymin, xmax, ymax)

    Args:
        bboxes: A numpy array containing the bounding box 8-coordinates.

    Returns:
        A list containing the bounding box 4-coordinates.
    """
    
    if len(bboxes) != 8:
        raise NotImplementedError("The bounding box coordinates must a length of 8!")
    
    Xs = bboxes[0::2]
    Ys = bboxes[1::2]
    
    xmin = int(round(np.min(Xs, 0)))
    ymin = int(round(np.min(Ys, 0)))
    xmax = int(round(np.max(Xs, 0)))
    ymax = int(round(np.max(Ys, 0)))
    
    final_boxes = [xmin, ymin, xmax, ymax]
    
    return final_boxes


# Taken from https://gist.github.com/flashlib/e8261539915426866ae910d55a3f9959
def order_point_clockwise(bboxes: np.ndarray) -> np.ndarray:
    """
    Order in clockwise the bounding box coordinates.

    Args:
        bboxes: A numpy array containing the bounding box coordinates. Shape: [4, 2].

    Returns:
        An ordered clockwise bounding box.

    """
    if bboxes.ndim != 2 and bboxes.shape != (4, 2):
        raise ValueError("The bounding box coordinates are not in the correct shape!"
                         "It must be an numpy array of 2D whose shape is (4, 2).")
    
    # sort the points based on their x-coordinates
    xSorted = bboxes[np.argsort(bboxes[:, 0]), :]
    
    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (top_left, bottom_left) = leftMost
    
    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (top_right, bottom_right) = rightMost
    
    # return the coordinates in this following order: top-left, top-right, bottom-right, and bottom-left
    return np.array([top_left, top_right, bottom_right, bottom_left])
