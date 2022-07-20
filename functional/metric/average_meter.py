from typing import Optional


# Adapted from https://github.com/pytorch/examples/blob/adc5bb40f1fa5ebae690787b474af4619df170b8/imagenet/main.py#L363
class AverageMeter(object):

    def __init__(self, fmt: Optional[str] = ":f"):
        """
        Computes and stores the average and current value.

        Args:
            fmt (string, optional): The string format.

        """
        self.fmt: str = fmt

        self.count: int = 0

        self.value: float = 0.0

        self.total: float = 0.0

        self.global_avg: float = 0.0

    def update(self, value: float, n: Optional[int] = 1):
        """
        Update the current values.

        Args:
            value (float): The value to update with.
            n (int): A multiplier.

        """
        self.value = value
        self.total += (value * n)
        self.count += n
        self.global_avg = self.total / self.count

    def __str__(self):
        fmtstr = '{value' + self.fmt + '} ({global_avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
