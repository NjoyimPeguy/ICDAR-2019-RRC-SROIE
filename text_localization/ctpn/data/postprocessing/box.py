import numpy as np

from typing import Tuple


def decode(predicted_bboxes: np.ndarray, anchor_boxes: np.ndarray):
    """
    Decode the predicted bounding boxes.

    Args:
        predicted_bboxes: The predicted set of bounding boxes.
        anchor_boxes: The set of default boxes.

    Returns:
        The decoded bounding boxes that was predicted by the CTPN.

    """
    
    # The height of the anchor boxes.
    ha = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1.
    
    # The center y-axis of the anchor boxes.
    Cya = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2.
    
    # The center y-axis of the predicted boxes
    Vcy = predicted_bboxes[..., 1] * ha + Cya
    
    # The height of the predicted boxes
    Vhx = np.exp(predicted_bboxes[..., 3]) * ha
    
    x1 = anchor_boxes[:, 0]
    y1 = Vcy - Vhx / 2.
    x2 = anchor_boxes[:, 2]
    y2 = Vcy + Vhx / 2.
    
    bboxes = np.stack([x1, y1.squeeze(), x2, y2.squeeze()], axis=1)
    
    return bboxes


def clip_bboxes(bboxes: np.ndarray, image_size: Tuple[int, int]):
    """
    Clip the bounding boxes within the image boundary.

    Args:
        bboxes: The set of bounding boxes.
        image_size: The image's size.

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
        original_image: The original image.
        detected_bboxes: The bounding boxes detected inside the image.

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
            # then this bounding box is considered not empty. Otherwise it is empty.
            if avg_white_pixels < 250:
                qualified_bboxes.append(i)
    
    qualified_bboxes = np.array(qualified_bboxes, dtype=np.int32)
    
    return qualified_bboxes
