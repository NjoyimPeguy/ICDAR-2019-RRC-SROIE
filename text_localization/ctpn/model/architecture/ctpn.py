import math
import torch
import torchvision.models as models

from typing import Tuple
from pytorch_model_summary import summary


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 padding: Tuple[int, int],
                 use_activation: bool = True):
        """
        Implementation of a basic Conv-2D block.
        
        Args:
            in_channels: The number of the input channels.
            out_channels: The number of the output channels.
            kernel_size: A tuple of 2 integers, specifying the height and width of the 2D convolution window.
            stride: The number of pixels by which the window moves (on the height and width) after each operation.
            padding: The process of adding P zeroes to each side of the boundaries of the input
            use_activation: A function that will help to add non-linearity in the Network.
            
        """
        super(BasicBlock, self).__init__()
        
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, bias=True)
        self.activation = torch.nn.ReLU(inplace=True) if use_activation else None
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv(inputs)
        if self.activation is not None:
            output = self.activation(output)
        return output


class CTPN(torch.nn.Module):
    def __init__(self, f_out_channels: int, n_classes: int, n_anchors: int, pretrained_backbone: bool = False):
        """
        An implementation of the Connectionist Text Proposal Network.
        For further info about the model, check the paper: https://arxiv.org/abs/1609.03605
        
        Args:
            f_out_channels: The very first output channel.
            n_classes: The number of classes.
            n_anchors: The number of anchor boxes.
            pretrained_backbone: A boolean indicating whether the VGG16's model is loaded from scratch or with pretrained weights.
            
        """
        
        super(CTPN, self).__init__()
        
        self.hidden_size = 128
        self.n_classes = n_classes
        self.pretrained_backbone = pretrained_backbone
        
        # The VGG16's model.
        base_model = models.vgg16(pretrained=pretrained_backbone)
        layers = list(base_model.features)[:-1]
        self.base_layers = torch.nn.Sequential(*layers)
        
        # The first basic Conv2d layer after the last convolution maps (Conv5) of the VGG16.
        self.rpn = BasicBlock(in_channels=512,
                              out_channels=f_out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1))
        
        # The bi-directional LSTM.
        self.bilstm = torch.nn.LSTM(input_size=f_out_channels,
                                    hidden_size=self.hidden_size,
                                    num_layers=2,
                                    bidirectional=True,
                                    batch_first=True)
        
        # An equivalent of the fully-connected layer.
        self.fc = torch.nn.Linear(self.hidden_size * 2, f_out_channels, bias=True)
        
        # The classification layer.
        self.classification_layer = BasicBlock(in_channels=f_out_channels,
                                               out_channels=n_anchors * n_classes,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1),
                                               use_activation=False)
        
        # The regression layer.
        self.regression_layer = BasicBlock(in_channels=f_out_channels,
                                           out_channels=n_anchors * 4,
                                           kernel_size=(3, 3),
                                           stride=(1, 1),
                                           padding=(1, 1),
                                           use_activation=False)
        
        # In this layer, the output from regression layer will be concatenated to the classification_layer.
        # This will feed the classification layer with regressed location,
        # helping the classification layer to get better predictions.
        self.concatenated_classification_layer = BasicBlock(in_channels=n_anchors * (4 + n_classes),
                                                            out_channels=n_anchors * n_classes,
                                                            kernel_size=(3, 3),
                                                            stride=(1, 1),
                                                            padding=(1, 1),
                                                            use_activation=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        for name, module in self.named_children():
            if name == "bilstm":
                # init the Bi-LSTM layer based on this paper https://arxiv.org/abs/1702.00071
                # if len(param.shape) >= 2, then it is 'the weight_ih_l{}{}' or 'weight_hh_l{}{}'
                # Otherwise it is the 'bias_ih_l{}{}', 'bias_hh_l{}{}'
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        torch.nn.init.zeros_(param.data)
                        torch.nn.init.ones_(param.data[self.hidden_size:self.hidden_size * 2])
            elif not self.pretrained_backbone or name != "base_layers":
                for layer in module.modules():
                    if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                        relu_gain = torch.nn.init.calculate_gain(nonlinearity="relu", param=None)
                        torch.nn.init.kaiming_uniform_(layer.weight, a=relu_gain, mode="fan_in", nonlinearity="relu")
                        if layer.bias is not None:
                            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
                            bound = 1 / math.sqrt(fan_in)
                            torch.nn.init.uniform_(layer.bias, -bound, bound)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        backbone_features = self.base_layers(inputs)
        
        rpn_features = self.rpn(backbone_features)  # Shape: [B, C, H, W]
        
        rpn_size = (rpn_features.size(0), rpn_features.size(1), rpn_features.size(2), rpn_features.size(3))
        
        rpn_features = rpn_features.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        
        rpn_features = rpn_features.contiguous().view(-1, rpn_features.size(2), rpn_features.size(3))
        
        bilstm_features, _ = self.bilstm(rpn_features)  # Shape: [B x H, W, C]
        
        fc_features = self.fc(bilstm_features)  # Shape: [B x H, W, C]
        
        # Shape: [B, H, W, C]
        fc_features = fc_features.contiguous().view(rpn_size[0], rpn_size[2], rpn_size[3], rpn_size[1])
        
        fc_features = fc_features.permute(0, 3, 1, 2).contiguous()  # Shape: [B, C, H, W]
        
        regression_features = self.regression_layer(fc_features)  # Shape: [B, C, H, W]
        classification_features = self.classification_layer(fc_features)  # Shape: [B, C, H, W]
        
        # Concatenating the output of the regression layer into the classification layer.
        regr_ = regression_features.detach()
        concat = torch.cat([classification_features, regr_], dim=1)
        concat_cls_features = self.concatenated_classification_layer(concat)
        
        regression_features = regression_features.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        concat_cls_features = concat_cls_features.permute(0, 2, 3, 1)  # Shape: [B, H, W, C]
        
        # Shape. [B, N, 4]
        regression_features = regression_features.contiguous().view(regression_features.size(0), -1, 4)
        
        # Shape. [B, N, 2]
        concat_cls_features = concat_cls_features.contiguous().view(concat_cls_features.size(0), -1, self.n_classes)
        
        output = (regression_features, concat_cls_features)
        
        return output


if __name__ == '__main__':
    args = dict([
        ["f_out_channels", 256],
        ["n_classes", 2],
        ["n_anchors", 10],
        ["pretrained_backbone", True]
    ])
    model = CTPN(**args)
    
    B, C, H, W = 10, 3, 2048, 1024
    image = torch.randn(size=(B, C, H, W))
    
    model_summary = summary(model, image, max_depth=50, show_input=True)
    print(model_summary)
