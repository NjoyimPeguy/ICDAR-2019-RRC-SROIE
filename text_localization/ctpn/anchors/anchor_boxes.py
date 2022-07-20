import numpy as np

from typing import List


def generate_basic_anchors(anchor_heights: List[float], anchor_shift: int) -> np.ndarray:
    """
    Generate the basic anchor boxes (their relative coordinates) based on the provided anchor heights.
    
    Args:
        anchor_heights (float, list): A list containing the height of the anchor boxes.
        anchor_shift (float, list): The width of each anchor box.

    Returns:
        A numpy array whose shape is (N, 4) where N is the number of anchor heights and
        contains the coordinates of the basic anchor boxes.
        
    """
    basic_anchor: np.ndarray = np.array([0, 0, anchor_shift - 1, anchor_shift - 1], np.float32)

    heights: np.ndarray = np.array(anchor_heights, dtype=np.float32)

    widths: np.ndarray = np.ones(len(heights), dtype=np.float32) * anchor_shift

    sizes: np.ndarray = np.column_stack((heights, widths))

    basic_anchors: np.ndarray = np.apply_along_axis(func1d=scale_anchor, axis=1, arr=sizes, basic_anchor=basic_anchor)

    return basic_anchors


def scale_anchor(shape: np.ndarray, basic_anchor: np.ndarray) -> np.ndarray:
    """
    Scale anchor boxes based on the widths and heights.
    
    Args:
        shape (numpy.ndarray): A numpy array containing the shape of the anchor box.
        basic_anchor (numpy.ndarray): A numpy array containing the coordinates of the anchor box.

    Returns:
        A numpy array containing the coordinates of the anchor box.

    """

    h, w = shape

    cx: float = (basic_anchor[0] + basic_anchor[2]) / 2.0

    cy: float = (basic_anchor[1] + basic_anchor[3]) / 2.0

    scaled_anchor: np.ndarray = basic_anchor.copy()

    scaled_anchor[0] = cx - w / 2.0  # xmin
    scaled_anchor[1] = cy - h / 2.0  # ymin
    scaled_anchor[2] = cx + w / 2.0  # xmax
    scaled_anchor[3] = cy + h / 2.0  # ymax

    return scaled_anchor


def generate_all_anchor_boxes(feature_map_size: List[float],
                              feat_stride: int,
                              anchor_heights: List[float],
                              anchor_shift: int) -> np.ndarray:
    """
    Generate all anchors corresponding to a feature map generated by a CNN network.
    
    Args:
        feature_map_size (float, list): A list containing the size of the feature map.
        feat_stride (int): The stride of the feature map.
        anchor_heights (float, list): A list containing the height of the anchor boxes.
        anchor_shift (int): The width of each anchor box.

    Returns:
        A numpy array whose shape is [#anchors, 4] and contains the coordinates of the generated anchor boxes.
        
    """

    # Generate basic anchor boxes
    basic_anchors: np.ndarray = generate_basic_anchors(anchor_heights, anchor_shift)

    n_anchors: int = basic_anchors.shape[0]

    feat_map_h, feat_map_w = feature_map_size

    all_anchors: np.ndarray = np.zeros(shape=(n_anchors * feat_map_h * feat_map_w, 4), dtype=np.float32)

    # Compute and return all anchor boxes on the feature maps.
    index = 0
    for y in range(feat_map_h):
        for x in range(feat_map_w):
            shift = np.array([x, y, x, y]) * feat_stride
            all_anchors[index:index + n_anchors, :] = basic_anchors + shift
            index += n_anchors

    return all_anchors