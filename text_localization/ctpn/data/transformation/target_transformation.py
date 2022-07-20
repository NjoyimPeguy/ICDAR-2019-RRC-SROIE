import torch
import numpy as np

from torch import Tensor
from typing import Tuple, Optional
from text_localization.ctpn.anchors import generate_all_anchor_boxes


class TargetTransform(object):

    def __init__(self, configs: dict):
        """
        Perform the bounding box transformations.

        Args:
            configs (dict): The configuration file.
        """
        self.configs: dict = configs

    def __call__(self, gt_boxes: Tensor, image_size: Tuple[int, int], return_anchor_boxes: Optional[bool] = False):
        """

        Args:
            gt_boxes (Tensor): The set of ground truth boxes.
            image_size (int, tuple): The image size.
            return_anchor_boxes (bool, optional): A boolean indicating whether to return the set of anchor boxes or not.

        Returns:
            A tuple containing the encoded ground truth boxes and ground truth labels,
            ground truth boxes if specified and anchor boxes if specified.
        """
        h, w = image_size

        anchor_shift = self.configs.ANCHOR.SHIFT

        # Estimate the size of feature map created by Convolutional neural network (VGG-16)
        feature_map_size = [int(np.ceil(h / anchor_shift)), int(np.ceil(w / anchor_shift))]

        anchor_boxes = generate_all_anchor_boxes(
            feature_map_size=feature_map_size,
            feat_stride=self.configs.IMAGE.FEAT_STRIDE,
            anchor_heights=self.configs.ANCHOR.HEIGHTS,
            anchor_shift=anchor_shift
        )

        matches = match_anchor_boxes(
            configs=self.configs,
            image_size=image_size,
            anchor_boxes=torch.as_tensor(anchor_boxes, device=torch.device("cpu")),
            gt_boxes=gt_boxes
        )

        if return_anchor_boxes:
            return matches + (anchor_boxes,)

        return matches


def compute_intersection(anchor_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    """
    Compute the intersection between each and every two set of bounding boxes.

    Args:
        anchor_boxes (Tensor): The set of default boxes. Shape: [M, 1, 4]
        gt_boxes (Tensor): The set of ground truth boxes. Shape: [1, N, 4].

    Returns:
        The intersection between ground truth and anchor boxes. Shape: [M, N].

    """
    overlaps_top_left = torch.maximum(anchor_boxes[..., :2], gt_boxes[..., :2])  # Shape: [M, N, 2]

    overlap_bottom_right = torch.minimum(anchor_boxes[..., 2:], gt_boxes[..., 2:])  # Shape: [M, N, 2]

    diff = overlap_bottom_right - overlaps_top_left

    max_ = torch.maximum(diff, torch.as_tensor(0.0, device=gt_boxes.device))  # Shape: [M, N, 2]

    intersection = max_[..., 0] * max_[..., 1]  # Shape: [M, N]

    return intersection


def jaccard_index(anchor_boxes: Tensor, gt_boxes: Tensor, eps: Optional[float] = 1e-6) -> Tensor:
    """
    Compute the IoU between each and every two sets of bounding boxes.

    Args:
        anchor_boxes (Tensor): The anchor box coordinates. Shape: [M, 1, 4].
        gt_boxes (Tensor): The ground truth coordinates. Shape: [1, N, 4].
        eps (float, optional): a small number to avoid 0 as denominator.

    Returns:
        The Jaccard index/overlap between ground truth and anchor boxes. Shape: [M, N].

    """

    # Computing the intersection between two sets of bounding boxes.
    intersection = compute_intersection(anchor_boxes=anchor_boxes, gt_boxes=gt_boxes)

    # Computing the area of each bounding box in the set of anchor boxes.
    # Area shape: [M, 1]
    anchor_box_areas = (anchor_boxes[..., 2] - anchor_boxes[..., 0] + 1.) * \
                       (anchor_boxes[..., 3] - anchor_boxes[..., 1] + 1.)

    # Computing the area of each bounding box in the set of ground truth boxes.
    # Area shape: [1, N]
    gt_box_areas = (gt_boxes[..., 2] - gt_boxes[..., 0] + 1.) * \
                   (gt_boxes[..., 3] - gt_boxes[..., 1] + 1.)

    union_area = anchor_box_areas + gt_box_areas - intersection  # Shape: [M, N]

    IoU = intersection / (union_area + eps)  # Shape: [M, N]

    return IoU


def match_anchor_boxes(configs: dict, image_size: Tuple[int, int], anchor_boxes: Tensor, gt_boxes: Tensor):
    """

    Match default/prior/anchor boxes to any ground truth with jaccard overlap higher than a certain threshold.

    Args:
        configs (dict): The config path_to_file.
        image_size (int, tuple): The image's size.
        anchor_boxes (Tensor): The set of anchor boxes. Shape: [M, 4], where 'M' is the number of anchor boxes.
        gt_boxes (Tensor): The set of ground truth boxes. Shape: [N, 4], where 'N' is the number of ground truth boxes.

    Returns:
        The encoded bounding boxes and labels.

    """

    # Useful variables for anchor matching.
    ignore_index = configs.ANCHOR.IGNORE_LABEL

    positive_anchor_label = configs.ANCHOR.POSITIVE_LABEL

    negative_anchor_label = configs.ANCHOR.NEGATIVE_LABEL

    positive_jaccard_overlap_threshold = configs.RPN.POSITIVE_JACCARD_OVERLAP

    negative_jaccard_overlap_threshold = configs.RPN.NEGATIVE_JACCARD_OVERLAP

    # Compute the IoU between anchor and ground truth boxes. # Shape: [M, N]
    IoUs = jaccard_index(anchor_boxes=torch.unsqueeze(anchor_boxes, dim=1), gt_boxes=torch.unsqueeze(gt_boxes, dim=0))

    device = gt_boxes.device

    n_gt_boxes = IoUs.size(1)

    # Declaration and initialisation of a new tensor containing the binary label for each anchor box.
    # For text/non-text classification, a binary label is assigned to each positive (text) or
    # negative (non-text) anchor. It is defined by computing the IoU overlap with the GT bounding box.
    # For now, We do not care about positive/negatives anchors.
    anchor_labels = torch.full(size=(anchor_boxes.shape[0],), fill_value=ignore_index, dtype=torch.int64)

    _, best_anchor_for_each_target_index = torch.max(IoUs, dim=0)

    best_target_for_each_anchor, best_target_for_each_anchor_index = torch.max(IoUs, dim=1)

    # Assigning each GT box to the corresponding maximum-overlap-anchor.
    best_target_for_each_anchor_index[best_anchor_for_each_target_index] = torch.arange(n_gt_boxes, device=device)

    # Ensuring that every GT box has an anchor assigned.
    best_target_for_each_anchor[best_anchor_for_each_target_index] = positive_anchor_label

    # Taking the real labels for each anchor.
    anchor_labels = anchor_labels[best_target_for_each_anchor_index]

    # A positive anchor is defined as : an anchor that has an > IoU overlap threshold with any GT box;
    anchor_labels[best_target_for_each_anchor > positive_jaccard_overlap_threshold] = positive_anchor_label

    # The negative anchors are defined as < IoU overlap threshold with all GT boxes.
    anchor_labels[best_target_for_each_anchor < negative_jaccard_overlap_threshold] = negative_anchor_label

    # Finally, we ignore anchor boxes that are outside the image.
    img_h, img_w = image_size
    outside_anchors = torch.where(
        (anchor_boxes[:, 0] < 0) |
        (anchor_boxes[:, 1] < 0) |
        (anchor_boxes[:, 2] > img_w) |
        (anchor_boxes[:, 3] > img_h)
    )[0]
    anchor_labels[outside_anchors] = ignore_index

    # calculate bounding box targets.
    matched_gt_bboxes = gt_boxes[best_target_for_each_anchor_index]
    bbox_targets = encode(matched_gt_bboxes, anchor_boxes)

    output = (bbox_targets, anchor_labels)

    return output


def encode(gt_boxes: Tensor, anchor_boxes: Tensor):
    """
    Compute relative predicted vertical coordinates (v) with respect to the bounding box location of an anchor.

    Args:
        gt_boxes (Tensor): The ground truth coordinates.
        anchor_boxes (Tensor): The anchor box coordinates.

    Returns:
        The relative predicted vertical coordinates in center form.

    """

    # The height of the ground truth boxes.
    h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.

    # The height of the anchor boxes
    ha = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1.

    # The center y-axis of the ground truth boxes
    Cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

    # The center y-axis of the anchor boxes
    Cya = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2.0

    Vc = (Cy - Cya) / ha

    Vh = torch.log(h / ha)

    bboxes = torch.stack([Vc, Vh], dim=1)

    return bboxes
