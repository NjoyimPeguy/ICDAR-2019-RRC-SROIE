import math
import numpy as np
import torch.nn.functional as F

from .non_max_suppression import nms
from .box import decode, clip_bboxes
from text_localization.ctpn.model.anchors import generate_all_anchor_boxes
from text_localization.ctpn.model.text_connector import TextProposalConnector


class TextDetector:
    def __init__(self, configs: dict):
        """
        Detect text in a image.
        
        Args:
            configs: The config file.
            
        """
        self.configs = configs
        self.text_proposal_connector = TextProposalConnector(configs)
    
    def __call__(self, predictions, image_size):
        """
        Perform the text localization.
        
        Args:
            predictions: The model's predictions.
            image_size: The image's size.

        Returns:
            A tuple containing the predicted bounding boxes and scores.
            
        """
        h, w = image_size
        
        predicted_bboxes, predicted_scores = predictions
        
        predicted_scores = F.softmax(predicted_scores, dim=2)
        
        # Putting all to numpy array
        predicted_bboxes = predicted_bboxes.detach().cpu().numpy()
        predicted_scores = predicted_scores.detach().cpu().numpy()
        
        anchor_scale = self.configs.ANCHOR.SCALE
        
        # Estimate the size of feature map created by Convolutional neural network (VGG-16)
        feature_map_size = [int(math.ceil(h / anchor_scale)), int(math.ceil(w / anchor_scale))]
        
        # Generate all anchor boxes.
        anchor_boxes = generate_all_anchor_boxes(
            feature_map_size=feature_map_size,
            feat_stride=self.configs.IMAGE.FEAT_STRIDE,
            anchor_heights=self.configs.ANCHOR.HEIGHTS,
            anchor_scale=anchor_scale
        )
        
        # Decoding the model's predictions.
        decoded_bboxes = decode(predicted_bboxes=predicted_bboxes, anchor_boxes=anchor_boxes)
        
        # Keeping the predicted/decoded boxes inside the image.
        clipped_bboxes = clip_bboxes(bboxes=decoded_bboxes, image_size=image_size)
        
        # Taking only boxes and scores based on the text proposal minimum score.
        class_idx = self.configs.ANCHOR.POSITIVE_LABEL  # 0:background, 1:text
        conf_scores = predicted_scores[0, :, class_idx]
        conf_scores_mask = np.where(conf_scores > self.configs.TEXT_PROPOSALS.MIN_SCORE)[0]
        
        selected_bboxes = clipped_bboxes[conf_scores_mask, :]
        selected_scores = predicted_scores[0, conf_scores_mask, class_idx]
        
        # Perform the non-max-suppression to eliminate unnecessary bounding boxes.
        candidates = nms(bboxes=selected_bboxes,
                         scores=selected_scores,
                         IoU_threshold=self.configs.TEXT_PROPOSALS.NMS_THRESH)
        
        selected_bboxes, selected_scores = selected_bboxes[candidates], selected_scores[candidates]
        
        # Taking the text lines.
        text_lines, scores = self.text_proposal_connector.get_text_lines(text_proposals=selected_bboxes,
                                                                         scores=selected_scores,
                                                                         im_size=image_size)
        
        # detections: 0 = detected bboxes, detections: 1 = detected scores
        detections = (text_lines, scores)
        
        return detections
