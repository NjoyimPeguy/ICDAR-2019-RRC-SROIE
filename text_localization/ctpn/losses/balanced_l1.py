import torch

from torch import Tensor
from typing import Optional


class BalancedL1Loss(torch.nn.Module):

    def __init__(self,
                 alpha: Optional[float] = 0.5,
                 gamma: Optional[float] = 1.5,
                 beta: Optional[float] = 1.0,
                 reduction: Optional[str] = "none"):
        r"""
        An implementation of the Balanced L1 Loss as described in:

        `Libra R-CNN: Towards Balanced Learning for Object Detection <https://arxiv.org/abs/1904.02701>`__.

        It is computed as follows:

        .. math::
            L_{b(x)} = \begin{cases}
                        \frac{a}{b}(b|x| + 1)ln(b|x| + 1) - \alpha|x|, & \text{if } |x| < 1 \\
                        \gamma|x| + C, & \text{otherwise }
                        \end{cases}

            \text{in which the parameters \gamma, \alpha, and b are constrained by}

            \alpha ln(b + 1) = \gamma

        Shape:
            - Input: :math:`(N, *)`.
            - Target: :math:`(N, *)`.

            where '*' means any number of dimensions.

        Examples::

            >>> inputs = torch.randn(1, 1, 5, 5, requires_grad=True)
            >>> targets = torch.rand(1, 1, 5, 5, dtype=torch.float32)
            >>> l1Loss = BalancedL1Loss(reduction="mean")
            >>> loss = l1Loss(inputs, targets)
            >>> loss.backward()

        Args:
            alpha (string, optional): A float number that either increases more gradients for inliers (accurate samples)
                if it is small or increases for outliers (negative samples). By default, it is set to 0.5.
            gamma (float, optional): A float number that is used to tune the upper bound of regression errors.
            beta (float, optional): A float number that represents a threshold at which to change the loss behavior .
            reduction (string, optional): Specifies the reduction to apply to the
                 output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                 'mean': the sum of the output will be divided by the number of elements
                 in the output, 'sum': the output will be summed.
                 Default: 'none'.
            
        """

        super(BalancedL1Loss, self).__init__()

        reductions = ("none", "mean", "sum")
        if reduction not in reductions:
            raise NotImplemented("This reduction operation '{0}' is not currently supported! "
                                 "Try one of these operations: {1}".format(reduction, reductions))

        if beta <= 0:
            raise ValueError("This value of beta '{0}' must be strictly positive".format(beta))

        self.beta: float = beta

        self.alpha: float = alpha

        self.gamma: float = gamma

        self.reduction: str = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:

        if sorted(list(inputs.size())) != sorted(list(targets.size())):
            raise ValueError("Input and target dimensions does not match!")

        if inputs.dtype != targets.dtype:
            raise ValueError("The inputs and targets must have the same data type!")

        valid_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float16)
        if inputs.dtype not in valid_types:
            raise ValueError("The input data type must be one of these following types: {0}.".format(valid_types))

        if targets.dtype not in valid_types:
            raise ValueError("The target data type must be one of these following types: {0}.".format(valid_types))

        if not inputs.device == targets.device:
            raise ValueError("The input and target must be in the same device. "
                             "Got: input device = {0} and target device = {1}.".format(inputs.device, targets.device))

        if targets.numel() == 0:
            return inputs.sum() * 0

        x = torch.abs(inputs - targets)

        # parameters γ, α, and b are constrained by: αln(b + 1) = γ.
        b = torch.exp(torch.as_tensor(self.gamma / self.alpha)) - 1.0

        loss = torch.where(
            x < self.beta,
            self.alpha / b * (b * x + 1) * torch.log(b * x / self.beta + 1) - self.alpha * x,
            self.gamma * x + self.gamma / b - self.alpha * self.beta
        )

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass

        return loss
