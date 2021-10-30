import math
import torch

from typing import Tuple
from text_localization.ctpn.model.anchors import generate_all_anchor_boxes


class TargetTransform(object):
    
    def __init__(self, configs):
        self.configs = configs
    
    def __call__(self, gt_boxes, image_size):
        h, w = image_size
        anchor_scale = self.configs.ANCHOR.SCALE
        
        # Estimate the size of feature map created by Convolutional neural network (VGG-16)
        feature_map_size = [int(math.ceil(h / anchor_scale)), int(math.ceil(w / anchor_scale))]
        
        anchor_boxes = generate_all_anchor_boxes(
            feature_map_size=feature_map_size,
            feat_stride=self.configs.IMAGE.FEAT_STRIDE,
            anchor_heights=self.configs.ANCHOR.HEIGHTS,
            anchor_scale=anchor_scale
        )
        
        anchor_boxes = torch.as_tensor(anchor_boxes, device=gt_boxes.device)
        
        encoded_gt_boxes, gt_matched_labels = match_anchor_boxes(
            configs=self.configs,
            gt_boxes=gt_boxes,
            anchor_boxes=anchor_boxes,
            image_size=(h, w)
        )
        
        output = (encoded_gt_boxes, gt_matched_labels)
        
        return output, anchor_boxes


def compute_intersection(gt_boxes: torch.Tensor, anchor_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute the intersection between each and every two set of bounding boxes.

    Args:
        gt_boxes: The set of ground truth boxes. Shape: [1, N, 4].
        anchor_boxes: The set of default boxes. Shape: [M, 1, 4]

    Returns:
        The intersection between ground truth and anchor boxes. Shape: [M, N].

    """
    overlaps_top_left = torch.maximum(gt_boxes[..., :2], anchor_boxes[..., :2])  # Shape: [M, N, 2]
    overlap_bottom_right = torch.minimum(gt_boxes[..., 2:], anchor_boxes[..., 2:])  # Shape: [M, N, 2]
    
    diff = overlap_bottom_right - overlaps_top_left
    
    max_ = torch.maximum(diff, torch.as_tensor(0.0, device=gt_boxes.device))  # Shape: [M, N, 2]
    
    intersection = max_[..., 0] * max_[..., 1]  # Shape: [M, N]
    
    return intersection


def jaccard_index(gt_boxes: torch.Tensor, anchor_boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the IoU between each and every two sets of bounding boxes.

    Args:
        gt_boxes: The ground truth coordinates. Shape: [1, N, 4].
        anchor_boxes: The anchor box coordinates. Shape: [M, 1, 4].
        eps: a small number to avoid 0 as denominator.

    Returns:
        The Jaccard index/overlap between ground truth and anchor boxes. Shape: [M, N].

    """
    
    # Computing the intersection between two sets of bounding boxes.
    intersection = compute_intersection(gt_boxes, anchor_boxes)
    
    # Computing the area of each bounding box in the set of ground truth boxes.
    # Area shape: [1, N]
    gt_box_areas = (gt_boxes[..., 2] - gt_boxes[..., 0] + 1.) * \
                   (gt_boxes[..., 3] - gt_boxes[..., 1] + 1.)
    
    # Computing the area of each bounding box in the set of anchor boxes.
    # Area shape: [M, 1]
    anchor_box_areas = (anchor_boxes[..., 2] - anchor_boxes[..., 0] + 1.) * \
                       (anchor_boxes[..., 3] - anchor_boxes[..., 1] + 1.)
    
    union_area = anchor_box_areas + gt_box_areas - intersection  # Shape: [M, N]
    
    IoU = intersection / (union_area + eps)  # Shape: [M, N]
    
    return IoU


def match_anchor_boxes(configs: dict, anchor_boxes: torch.Tensor, gt_boxes: torch.Tensor, image_size: Tuple[int, int]):
    """
    Match default/prior/anchor boxes to any ground truth with jaccard overlap higher than a certain threshold.
    Then encode the bounding boxes and return the matched/encoded bounding boxes and the anchor ID labels.

    Args:
        configs: The config file.
        anchor_boxes: The set of anchor boxes. Shape: [M, N], where 'M' is the number of anchor boxes.
        gt_boxes: The set of ground truth boxes. Shape: [N, 4].
        image_size: The image's size.

    Returns:
        The encoded bounding boxes and labels.

    """
    
    # Useful variables for anchor matching.
    ignore_index = configs.ANCHOR.IGNORE_LABEL
    positive_anchor_label = configs.ANCHOR.POSITIVE_LABEL
    negative_anchor_label = configs.ANCHOR.NEGATIVE_LABEL
    
    positive_jaccard_overlap_threshold = configs.RPN.POSITIVE_JACCARD_OVERLAP
    negative_jaccard_overlap_threshold = configs.RPN.NEGATIVE_JACCARD_OVERLAP
    
    # Compute the IoU between anchor and ground truth boxes.
    IoUs = jaccard_index(torch.unsqueeze(gt_boxes, dim=0), torch.unsqueeze(anchor_boxes, dim=1))  # Shape: [M, N]
    
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
    
    # calculate bbox targets
    bbox_targets = encode(gt_boxes[best_target_for_each_anchor_index], anchor_boxes)
    
    return bbox_targets, anchor_labels


def encode(gt_boxes, anchor_boxes):
    """
    Compute relative predicted vertical coordinates (v) with respect to the bounding box location of an anchor.

    Args:
        gt_boxes: The ground truth coordinates.
        anchor_boxes: The anchor box coordinates.

    Returns:
        The relative predicted vertical coordinates in center form.

    """
    
    # The width of the ground truth boxes.
    w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.
    
    # The height of the ground truth boxes.
    h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.
    
    # The width of the anchor boxes
    wa = anchor_boxes[:, 2] - anchor_boxes[:, 0] + 1.
    
    # The height of the anchor boxes
    ha = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1.
    
    # The center x-axis of the ground truth boxes
    Cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.
    
    # The center y-axis of the ground truth boxes
    Cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.
    
    # The center x-axis of the anchor boxes
    Cxa = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2.
    
    # The center y-axis of the anchor boxes
    Cya = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2.
    
    Hcx = (Cx - Cxa) / wa
    Vcy = (Cy - Cya) / ha
    Hw = torch.log(w / wa)
    Vh = torch.log(h / ha)
    
    bboxes = torch.stack([Hcx, Vcy, Hw, Vh], dim=1)
    
    return bboxes
