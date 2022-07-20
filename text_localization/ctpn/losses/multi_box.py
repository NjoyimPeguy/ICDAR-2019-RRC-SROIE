import os
import sys

sys.path.append(os.getcwd())

import torch

from torch import Tensor
from typing import Tuple
from text_localization.ctpn.losses.balanced_l1 import BalancedL1Loss


class MultiBoxLoss(torch.nn.Module):

    def __init__(self, configs: dict):
        r"""
        An implementation of the multi box loss derived from:

        `Scalable Object Detection using Deep Neural Networks <https://arxiv.org/abs/1312.2249>`__.

        Args:
            configs (dict): The configuration file.

        """
        super(MultiBoxLoss, self).__init__()

        self.lambda_reg: float = configs.LOSS.LAMBDA_REG

        self.lambda_cls: float = configs.LOSS.LAMBDA_CLS

        self.neg_pos_ratio: float = configs.LOSS.NEG_POS_RATIO

        self.ignore_index: int = configs.ANCHOR.IGNORE_LABEL

        self.positive_anchor_label: int = configs.ANCHOR.POSITIVE_LABEL

        self.negative_anchor_label: int = configs.ANCHOR.NEGATIVE_LABEL

        self.balancedL1Loss: torch.nn.Module = BalancedL1Loss(reduction="none")

        self.ceLoss: torch.nn.Module = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none")

    def forward(self, predictions: Tuple[Tensor, Tensor], targets: Tuple[Tensor, Tensor]) -> Tensor:
        # The predicted boxes and classifications whose shapes are respectively:
        # (batch_size, #anchors, 2) and (batch_size, #anchors, #classes)
        predicted_bboxes, predicted_classes = predictions

        # The encoded ground truth boxes and matching indicators whose shapes are respectively:
        # (batch_size, #anchors, 2) and (batch_size, #anchors).
        gt_bboxes, matching_indicators = targets

        # Identify anchors that are positives.
        positive_anchor_mask = matching_indicators == self.positive_anchor_label

        # Identify anchors that are negatives.
        negative_anchor_mask = matching_indicators == self.negative_anchor_label

        # ==================================================================================================
        # Localization loss = BalancedL1Loss(predicted_bboxes, gt_bboxes) is computed over positive anchors.
        # ==================================================================================================

        # Shape: (#matched_anchors, 2)
        balanced_l1_loss = self.balancedL1Loss(inputs=predicted_bboxes[positive_anchor_mask],
                                               targets=gt_bboxes[positive_anchor_mask])

        # As in the paper, 'Nv' is the total number of anchors used by the localization loss.
        Nv = balanced_l1_loss.size(0)

        localization_loss = (self.lambda_reg / Nv) * balanced_l1_loss.sum()

        # ===========================================================================================
        # Confidence loss = CrossEntropyLoss(predicted_classes, gt_classes) is computed over positive
        # and (hard) negative anchors.
        # ===========================================================================================

        # Useful variable.
        n_classes = predicted_classes.size(2)

        # The confidence loss over positive anchors.
        positive_confidence_loss = self.ceLoss(
            input=predicted_classes[positive_anchor_mask].contiguous().view(-1, n_classes),
            target=matching_indicators[positive_anchor_mask].contiguous().view(-1)
        )

        # The confidence loss over negative anchors.
        negative_confidence_loss = self.ceLoss(
            input=predicted_classes[negative_anchor_mask].contiguous().view(-1, n_classes),
            target=matching_indicators[negative_anchor_mask].contiguous().view(-1)
        )

        # ==========================================================================================================
        # Now, instead of using all the negative anchors, they are sorted using the highest (negative based anchors)
        # confidence loss and pick the top ones so that the ratio between the negative and positive ones
        # is at most 'neg_pos_ratio:1'.
        # ==========================================================================================================

        # The number of positive anchors.
        k_positive_anchors = positive_anchor_mask.long().sum()

        # The number of all negative anchors.
        all_negatives = negative_anchor_mask.long().sum()

        # The real number of negative anchors, i.e., the ones that respect the ratio mentioned above.
        k_negatives = k_positive_anchors * self.neg_pos_ratio

        # The real number of negative anchors, i.e., the ones that respect the ratio mentioned above.
        K = min(k_negatives, all_negatives)

        # Now we sort the negative anchors by using the highest confidence loss and pick the K-top ones.
        hard_negative_confidence_loss, _ = torch.topk(input=negative_confidence_loss,
                                                      k=int(K),
                                                      largest=True,
                                                      sorted=True)

        # As in the paper, 'Ns' is the total number of anchors used by the confidence loss.
        # That is to say, the number of positive and negative anchors.
        Ns = torch.count_nonzero(positive_confidence_loss) + torch.count_nonzero(hard_negative_confidence_loss)

        # The sum over positive anchors.
        # Shape: (batch_size,)
        cls_pos = positive_confidence_loss.sum()

        # The sum over negative anchors.
        # Shape: (batch_size,)
        cls_neg = hard_negative_confidence_loss.sum()

        # The confidence loss is the sum over positive and hard negatives anchors.
        confidence_loss = (self.lambda_cls / Ns) * (cls_pos + cls_neg)

        return localization_loss, confidence_loss


if __name__ == "__main__":
    from text_localization.ctpn.configs import configs

    torch.manual_seed(1)

    batch_size, n_anchors, n_classes = 1, 30, 2

    predictions = (torch.randn(size=(batch_size, n_anchors, n_classes), requires_grad=True),
                   torch.randn(size=(batch_size, n_anchors, n_classes), requires_grad=True))

    targets = (torch.randn(size=(batch_size, n_anchors, n_classes)),
               torch.randint(low=-1, high=2, size=(batch_size, n_anchors)))

    loss = MultiBoxLoss(configs)

    output = loss(predictions, targets)

    print(output)
