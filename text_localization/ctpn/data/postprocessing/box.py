import numpy as np


def decode(predicted_bboxes: np.ndarray, anchor_boxes: np.ndarray) -> np.ndarray:
    """
    Decode the predicted bounding boxes.

    Args:
        predicted_bboxes (numpy array): The predicted set of bounding boxes.
        anchor_boxes (numpy array): The set of default boxes.

    Returns:
        The decoded bounding boxes that was predicted by the CTPN.

    """

    # The height of the anchor boxes.
    ha = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1.0

    # The center y-axis of the anchor boxes.
    Cya = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2.0

    # The center y-axis of the predicted boxes
    Vcy = predicted_bboxes[..., 0] * ha + Cya

    # The height of the predicted boxes
    Vhx = np.exp(predicted_bboxes[..., 1]) * ha

    x1 = anchor_boxes[:, 0]
    y1 = Vcy - Vhx / 2.0
    x2 = anchor_boxes[:, 2]
    y2 = Vcy + Vhx / 2.0

    bboxes = np.stack([x1, y1.squeeze(), x2, y2.squeeze()], axis=1)

    return bboxes
