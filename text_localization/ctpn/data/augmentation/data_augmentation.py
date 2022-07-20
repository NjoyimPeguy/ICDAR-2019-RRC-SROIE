from typing import List, Tuple

from ..transformation import SplitBBoxes

from functional.data.transformation.computer_vision import Compose, Resize, ToTensor, Normalize


class BasicDataTransformation(object):

    def __init__(self, configs: dict):
        """
        The basic data augmentation used during prediction or evaluation.

        Args:
            configs (dict): The configuration file.
        """

        self.image_size: Tuple[int, int] = tuple(configs.IMAGE.SIZE)

        self.pixel_std: List[float] = configs.IMAGE.PIXEL_STD

        self.pixel_mean: List[float] = configs.IMAGE.PIXEL_MEAN

        self.augment: Compose = self.get_transformation()

    def get_transformation(self):
        """
        Returns:
            A composition of image transformations.
        """
        return Compose([
            Resize(image_size=self.image_size),
            ToTensor(),
            Normalize(mean=self.pixel_mean, std=self.pixel_std)
        ])

    def __call__(self, image, boxes=None):
        return self.augment(image, boxes)


class TrainDataTransformation(BasicDataTransformation):

    def __init__(self, configs):
        """
        The data augmentation used during training.

        Args:
            configs (dict): The configuration path_to_file
        """

        self.anchor_shift: int = configs.ANCHOR.SHIFT

        super(TrainDataTransformation, self).__init__(configs)

    def get_transformation(self):
        return Compose([
            Resize(image_size=self.image_size, resize_bboxes=True),
            SplitBBoxes(anchor_shift=self.anchor_shift),
            ToTensor(),
            Normalize(mean=self.pixel_mean, std=self.pixel_std)
        ])
