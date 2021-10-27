import torch

from .balanced_l1 import BalancedL1Loss


class MultiBoxLoss(torch.nn.Module):
    
    def __init__(self, configs: dict, neg_pos_ratio: int = 1, lambda_reg: float = 1.0, lambda_cls: float = 1.0):
        """
        An implementation of the multi box loss derived from https://arxiv.org/abs/1312.2249.
        
        Args:
            configs: The config file.
            neg_pos_ratio: The ratio between negative and positive anchor boxes.
            
        """
        super(MultiBoxLoss, self).__init__()
        
        self.lambda_reg = lambda_reg
        
        self.lambda_cls = lambda_cls
        
        self.neg_pos_ratio = neg_pos_ratio
        
        self.ignore_label = configs.ANCHOR.IGNORE_LABEL
        
        self.positive_anchor_label = configs.ANCHOR.POSITIVE_LABEL
        
        self.negative_anchor_label = configs.ANCHOR.NEGATIVE_LABEL
        
        self.balanced_l1_loss = BalancedL1Loss(reduction="sum")
        
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction="none")
    
    def forward(self, predictions, targets):
        # The predicted boxes and classifications whose shapes are respectively:
        # [batch_size, #anchors, 4] and [batch_size, #anchors, #classes]
        predicted_boxes, predicted_classes = predictions
        
        # The encoded ground truth boxes and classes/labels whose shapes are respectively:
        # [batch_size, #anchors, 4] and [batch_size, #anchors].
        gt_bboxes, gt_classes = targets
        
        n_classes = predicted_classes.size(2)
        
        # ===================================================================================================================
        # Localisation loss = BalancedL1Loss((predicted_locs, gt_locs) is computed only over positive (non-background) priors
        # ===================================================================================================================
        
        # Identify anchors that are positives for localization/regression.
        positive_anchor_mask = gt_classes == self.positive_anchor_label
        
        # First, we take the matched ground truth boxes w.r.t the corresponding anchor boxes.
        gt_matched_boxes = gt_bboxes[positive_anchor_mask].contiguous().view(-1, 4)
        predicted_matched_boxes = predicted_boxes[positive_anchor_mask].contiguous().view(-1, 4)
        
        # As in the paper, 'Nv' is the total number of anchors used by the localization loss.
        # i.e., the number of positive anchors.
        Nv = gt_matched_boxes.size(0)
        
        # If there are no matching, then we return 0.0 loss.
        if Nv == 0.0:
            localization_loss = torch.tensor(data=0, dtype=torch.float32, requires_grad=True)
            confidence_loss = torch.tensor(data=0, dtype=torch.float32, requires_grad=True)
            return localization_loss, confidence_loss
        
        # The localization loss, i.e., the robust L1 loss is computed only over positive (non-background) anchors.
        localization_loss = (self.lambda_reg / Nv) * self.balanced_l1_loss(inputs=predicted_matched_boxes,
                                                                           targets=gt_matched_boxes)
        
        # =========================================================================================================
        # Confidence loss = CrossEntropyLoss(predicted_cls, gt_cls) is computed over positive and negative anchors.
        # =========================================================================================================
        
        # We first compute the confidence loss over positive anchors.
        positive_gt_classes = gt_classes[positive_anchor_mask].contiguous().view(-1)
        
        positive_predicted_classes = predicted_classes[positive_anchor_mask].contiguous().view(-1, n_classes)
        
        positive_confidence_loss = self.cross_entropy(input=positive_predicted_classes, target=positive_gt_classes)
        
        # Next, we compute the confidence loss over (hard) negative anchors.
        # After the matching step, most of the anchors boxes are negatives,
        # especially when the number of possible anchor boxes is large.
        # This introduces a significant imbalance between the positive and negative training anchors.
        # Instead of using all the negative anchors, they are sorted using the highest confidence loss
        # for each anchor box and pick the top ones so that the ratio between the negatives and positives
        # is at most 'neg_pos_ratio:1'. By default the ratio is 3:1.
        
        # First, we identify anchors that are negatives.
        negative_anchor_mask = gt_classes == self.negative_anchor_label
        
        negative_gt_classes = gt_classes[negative_anchor_mask].view(-1)
        
        negative_predicted_classes = predicted_classes[negative_anchor_mask].view(-1, n_classes)
        
        negative_confidence_loss = self.cross_entropy(input=negative_predicted_classes, target=negative_gt_classes)
        
        # The total number of all negative anchors.
        all_negatives = negative_confidence_loss.size(0)
        
        # The number of positive anchors.
        k_positives = positive_confidence_loss.size(0)
        
        # The real number of negative anchors, i.e., the ones that respect the ratio mentioned above.
        k_negatives = k_positives * self.neg_pos_ratio
        
        # The K-highest confidence loss to pick.
        K = min(k_negatives, all_negatives)
        
        # Now we sort the negative anchors by using the highest confidence loss and pick the K-top ones.
        hard_negative_confidence_loss, _ = torch.topk(input=negative_confidence_loss, k=K, largest=True, sorted=True)
        
        # As in the paper, 'Ns' is the total number of anchors used by the confidence loss.
        # i.e., the number of positive and negative anchors.
        Ns = positive_confidence_loss.size(0) + hard_negative_confidence_loss.size(0)
        
        # The confidence loss is the sum over positive and negatives anchors.
        cls_pos = positive_confidence_loss.sum()
        cls_neg = hard_negative_confidence_loss.sum()
        confidence_loss = (self.lambda_cls / Ns) * (cls_pos + cls_neg)
        
        return localization_loss, confidence_loss
