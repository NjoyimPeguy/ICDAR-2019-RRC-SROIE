import cv2
import numpy as np

from typing import Tuple, List, Optional


def to_xy_min_max(bboxes: np.ndarray) -> List[int]:
    """
    Convert one bounding box whose form is: [x1, y1, x2, y2, x3, y3, x4, y4]
    into a box of form (xmin, ymin, xmax, ymax)

    Args:
        bboxes (numpy.ndarray): A numpy array containing the bounding box 8-coordinates.

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
        bboxes (numpy.ndarray): A numpy array containing the bounding box coordinates. Shape: [4, 2].

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
    # y-coordinates, so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (top_left, bottom_left) = leftMost

    # now, sort the right-most coordinates according to their
    # y-coordinates, so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (top_right, bottom_right) = rightMost

    # return the coordinates in this following order: top-left, top-right, bottom-right, and bottom-left
    return np.array([top_left, top_right, bottom_right, bottom_left])


def clip_bboxes(bboxes: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Clip the bounding boxes within the image boundary.

    Args:
        bboxes (numpy.ndarray): The set of bounding boxes.
        image_size (int, tuple): The image's size.

    Returns:
        THe bounding boxes that are within the image boundaries.

    """

    height, width = image_size

    zero = 0.0
    w_diff = width - 1.0
    h_diff = height - 1.0

    # x1 >= 0 and x2 < width
    bboxes[:, 0::2] = np.maximum(np.minimum(bboxes[:, 0::2], w_diff), zero)
    # y1 >= 0 and y2 < height
    bboxes[:, 1::2] = np.maximum(np.minimum(bboxes[:, 1::2], h_diff), zero)

    return bboxes


def remove_empty_boxes(original_image: np.ndarray, detected_bboxes: np.ndarray) -> np.ndarray:
    """
    Remove empty bounding boxes based on the quantity of the white pixels.

    Args:
        original_image (numpy.ndarray): The original image.
        detected_bboxes (numpy.ndarray): The bounding boxes detected inside the image.

    Returns:
        A numpy array that contains the indexes of the non-empty bounding boxes.

    """
    qualified_bboxes = []
    for i, bbox in enumerate(detected_bboxes):

        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        cropped_image = original_image[ymin:ymax, xmin:xmax, :]

        if cropped_image.size > 0:

            total_pixels = np.sum(cropped_image)

            avg_white_pixels = total_pixels / cropped_image.size

            # if the average of white pixels is under 250 of the cropped image above,
            # then this bounding box is considered not empty. Otherwise, it is empty.
            if avg_white_pixels < 250:
                qualified_bboxes.append(i)

    qualified_bboxes = np.array(qualified_bboxes, dtype=np.int32)

    return qualified_bboxes


def draw_single_box(image: np.ndarray,
                    box_coordinates: Tuple[float, float, float, float],
                    color: Optional[Tuple[int, int, int]] = (0, 255, 0),
                    thickness: Optional[int] = 2) -> np.ndarray:
    """
    Draw a rectangle given a bounding box coordinates.

    Args:
        image (numpy.ndarray): The image to draw on to.
        box_coordinates (float, tuple): The single bounding box coordinate.
        color (int, tuple): The rectangle color.
        thickness (int): The rectangle thickness.

    Returns:

    """
    xmin, ymin, xmax, ymax = list(map(int, box_coordinates))

    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)

    return image


def draw_bboxes(image: np.ndarray,
                bboxes: np.ndarray,
                color: Optional[Tuple[int, int, int]] = (0, 255, 0),
                thickness: Optional[int] = 2) -> np.ndarray:
    """
    Draw rectangles given a set of bounding box coordinates.

    Args:
        image (numpy.ndarray): The image to draw on to.
        bboxes (numpy.ndarray): The bounding box coordinates.
        color (int, tuple): The rectangle color.
        thickness (int): The rectangle thickness.

    Returns:
        An drawn image with rectangles.
    """
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        image = draw_single_box(image, (xmin, ymin, xmax, ymax), color=color, thickness=thickness)

    return image
