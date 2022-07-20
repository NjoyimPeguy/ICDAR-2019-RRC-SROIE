import os
import sys

sys.path.append(os.getcwd())

import torch

from pytorch_model_summary import summary
from text_localization.ctpn.model import CTPN

if __name__ == "__main__":
    args = dict([
        ["n_classes", 2],
        ["n_anchors", 10],
        ["hidden_size", 128],
        ["f_out_channels", 256],
        ["pretrained_backbone", True]
    ])

    model = CTPN(**args)

    N, C, H, W = 1, 3, 224, 224

    image = torch.randn(size=(N, C, H, W))

    model_summary = summary(model, image, max_depth=50)

    print(model_summary)
