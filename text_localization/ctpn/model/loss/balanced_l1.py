import torch


class BalancedL1Loss(torch.nn.Module):
    
    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, reduction="mean"):
        """
        Implementation of the Balanced L1 Loss. arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
        
        Args:
            alpha: A float number that either increases more gradients for inliers (accurate samples) if it is small
                or increases for outliers (negative samples). By default, it is set to 0.5.
            gamma: A float number that is used to tune the upper bound of regression errors.
            beta: A float number that represents a threshold at which to change the loss behavior .
            reduction: The reduction operation to apply to the loss.
            
        """
        super(BalancedL1Loss, self).__init__()
        reductions = ("none", "mean", "sum")
        if reduction not in reductions:
            raise NotImplemented("This reduction operation '{0}' is not currently supported! "
                                 "Try one of these operations: {1}".format(reduction, reductions))
        
        assert beta > 0
        
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        diff = torch.abs(inputs - targets)
        
        # parameters γ, α, and b are constrained by αln(b + 1) = γ.
        b = torch.exp(torch.as_tensor(self.gamma / self.alpha)) - 1
        
        balanced_l1_loss = torch.where(
            diff < self.beta,
            (self.alpha / b) * (b * diff + 1) * torch.log(b * (diff / self.beta) + 1) - (self.alpha * diff),
            (self.gamma * diff) + ((self.gamma / b) - (self.alpha * self.beta))
        )
        
        if self.reduction == "mean":
            balanced_l1_loss = torch.mean(balanced_l1_loss)
        elif self.reduction == "sum":
            balanced_l1_loss = torch.sum(balanced_l1_loss)
        return balanced_l1_loss
