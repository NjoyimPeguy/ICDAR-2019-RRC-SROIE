import math
import torch
from torchvision.models import vgg16

from torch import Tensor
from typing import Tuple, Optional


class RPNBlock(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 padding: Tuple[int, int]):
        """
        Implementation of a basic Conv-2D block.

        Args:
            in_channels (int): The number of the input channels.
            out_channels (int): The number of the output channels.
            kernel_size (int, tuple): A tuple of 2 integers, specifying the height
                and width of the 2D convolution window.
            stride (int, tuple): The number of pixels by which the window moves (on the height and width)
                after each operation.
            padding (int, tuple): The process of adding P zeroes to each side of the boundaries of the input

        """
        super(RPNBlock, self).__init__()

        self.rpn_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=True),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.rpn_block(inputs)
        return output


class CTPN(torch.nn.Module):
    def __init__(self,
                 n_classes: int,
                 f_out_channels: int,
                 n_anchors: int,
                 hidden_size: int,
                 pretrained_backbone: Optional[bool] = False):
        """
        An implementation of the Connectionist Text Proposal Network as described in:
        `Detecting Text in Natural Image with Connectionist Text Proposal Network
        <https://arxiv.org/abs/1609.03605>`__.

        Args:
            n_classes (int): The number of classes.
            f_out_channels (int): The very first output channel.
            n_anchors (int): The number of anchor boxes.
            hidden_size (int): The number of nodes in each bilstm layer.
            pretrained_backbone (bool, optional): A boolean indicating whether the backbone model is loaded from scratch
                or with pretrained weights.
        """

        super(CTPN, self).__init__()

        self.n_classes: int = n_classes

        self.hidden_size: int = hidden_size

        self.pretrained_backbone: bool = pretrained_backbone

        vgg16_layers = list(vgg16(pretrained=pretrained_backbone).features)[:-1]

        self.vgg16_layers: torch.nn.Sequential = torch.nn.Sequential(*vgg16_layers)

        # The first basic Conv2d layer after the last convolution maps (Conv5) of the VGG16.
        self.rpn_layer: torch.nn.Module = RPNBlock(in_channels=512,
                                                   out_channels=f_out_channels,
                                                   kernel_size=(3, 3),
                                                   stride=(1, 1),
                                                   padding=(1, 1))

        # The bidirectional LSTM.
        self.bilstm_layer: torch.nn.Module = torch.nn.LSTM(input_size=f_out_channels,
                                                           hidden_size=self.hidden_size,
                                                           num_layers=2,
                                                           bidirectional=True,
                                                           batch_first=True)

        # An equivalent of the fully-connected layer.
        self.fc_layer: torch.nn.Module = torch.nn.Linear(self.hidden_size * 2, f_out_channels, bias=True)

        # The regression layer.
        self.regression_layer: torch.nn.Module = torch.nn.Conv2d(in_channels=f_out_channels,
                                                                 out_channels=n_anchors * 2,
                                                                 kernel_size=(1, 5),
                                                                 stride=(1, 1),
                                                                 padding=(0, 2),
                                                                 bias=True)

        # The classification layer.
        self.classification_layer: torch.nn.Module = torch.nn.Conv2d(in_channels=f_out_channels,
                                                                     out_channels=n_anchors * n_classes,
                                                                     kernel_size=(1, 5),
                                                                     stride=(1, 1),
                                                                     padding=(0, 2),
                                                                     bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for name, named_child in self.named_children():
            if name == "bilstm_layer":
                # init the Bi-LSTM layer based on this paper https://arxiv.org/abs/1702.00071
                # if len(param.shape) >= 2, then it is 'the weight_ih_l{}{}' or 'weight_hh_l{}{}'
                # Otherwise it is the 'bias_ih_l{}{}', 'bias_hh_l{}{}'
                for param in named_child.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        torch.nn.init.zeros_(param.data)
                        torch.nn.init.ones_(param.data[self.hidden_size:self.hidden_size * 2])
            elif not self.pretrained_backbone or name != "vgg16_layers":
                for layer in named_child.modules():
                    if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                        # Initialization of the learnable parameters as described in:
                        # `Delving deep into rectifiers: Surpassing human-level
                        # performance on ImageNet classification` - He, K. et al. (2015)
                        # https://arxiv.org/abs/1502.01852
                        torch.nn.init.kaiming_normal_(layer.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                        if layer.bias is not None:
                            torch.nn.init.constant_(layer.bias, val=0.0)

    def forward(self, inputs: Tensor) -> Tensor:

        backbone_features = self.vgg16_layers(inputs)  # Shape: [B, C, H, W]

        rpn_features = self.rpn_layer(backbone_features)  # Shape: [B, C, H, W]

        rpn_size = (rpn_features.size(0), rpn_features.size(1), rpn_features.size(2), rpn_features.size(3))

        rpn_features = rpn_features.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        rpn_features = rpn_features.contiguous().view(-1, rpn_features.size(2), rpn_features.size(3))

        bilstm_features, _ = self.bilstm_layer(rpn_features)  # Shape: [B x H, W, C]

        fc_features = self.fc_layer(bilstm_features)  # Shape: [B x H, W, C]

        # Shape: [B, H, W, C]
        fc_features = fc_features.contiguous().view(rpn_size[0], rpn_size[2], rpn_size[3], rpn_size[1])

        fc_features = fc_features.permute(0, 3, 1, 2).contiguous()  # Shape: [B, C, H, W]

        regression_features = self.regression_layer(fc_features)  # Shape: [B, C, H, W]

        classification_features = self.classification_layer(fc_features)  # Shape: [B, C, H, W]

        regression_features = regression_features.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]

        # Shape: [B, H, W, C]
        classification_features = classification_features.permute(0, 2, 3, 1)

        # Shape. [B, N, 4]
        regression_features = regression_features.contiguous().view(regression_features.size(0), -1, 2)

        # Shape. [B, N, 2]
        classification_features = classification_features.contiguous().view(
            classification_features.size(0), -1, self.n_classes
        )

        output = (regression_features, classification_features)

        return output
