import numpy as np

from typing import List


def nms(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Compute the non max-suppression algorithm.

    Args:
        bboxes (numpy.ndarray): The bounding box coordinates.
        scores (numpy.ndarray): The scores for each bounding box coordinate.
        iou_threshold (float): The Jaccard overlap threshold.

    Returns:
        A list containing the best indices out of a set of overlapping bounding boxes.

    """

    # Grabbing the coordinates of the bounding boxes
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    areas = (xmax - xmin + 1) * (ymax - ymin + 1)

    # Sorting the scores in descending order
    score_indices = np.argsort(scores, kind="mergesort", axis=-1)[::-1]

    zero = 0.0

    candidates = []

    while score_indices.size > 0:
        # Picking the index of the highest IoU
        i = score_indices[0]

        candidates.append(i)

        # Finding the highest (xmin, ymin) coordinates
        xxmax = np.maximum(xmin[i], xmin[score_indices[1:]])
        yymax = np.maximum(ymin[i], ymin[score_indices[1:]])

        # Finding the smallest (xmax, ymax) coordinates
        xxmin = np.minimum(xmax[i], xmax[score_indices[1:]])
        yymin = np.minimum(ymax[i], ymax[score_indices[1:]])

        # compute the width and height of the bounding box
        w = np.maximum(zero, xxmin - xxmax)
        h = np.maximum(zero, yymin - yymax)

        area_of_overlap = w * h
        remaining_areas = areas[score_indices[1:]]
        area_of_union = areas[i] + remaining_areas - area_of_overlap

        # Computing the Intersection Over Union. That is:
        # dividing the area of overlap between the bounding boxes by the area of union
        IoU = area_of_overlap / area_of_union

        # Keeping only elements with an IoU <= thresh
        indices = np.where(IoU <= iou_threshold)[0]
        score_indices = score_indices[indices + 1]

    return candidates
