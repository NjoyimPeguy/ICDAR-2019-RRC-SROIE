import cv2

# Apparently, using CV2 with dataloader where num_workers > 0
# you will encounter a deadlock. This is because of CV2 using multithreading.
# The solution is to set the number of threads to zero.
# For further info, check this out:
# https://stackoverflow.com/questions/54013846/pytorch-dataloader-stucked-if-using-opencv-resize-method
cv2.setNumThreads(0)

import torch
import numpy as np

from scipy.stats import norm
from .base import AbstractTransformation
from typing import Tuple, List, Union, Optional

random_state = np.random.RandomState()


class Compose(AbstractTransformation):

    def __init__(self, transformations: List[AbstractTransformation]):
        """
        Composes several transformation together.

        Args:
            transformations (list):  List of transformations to compose.
        """

        self.transformations: List[AbstractTransformation] = transformations

    def apply(self, image: np.ndarray, bboxes: Union[np.ndarray, None] = None):
        for transformation in self.transformations:
            image, bboxes = transformation(image, bboxes)

        return image, bboxes


class Resize(AbstractTransformation):

    def __init__(self,
                 image_size: Tuple[int, int],
                 resize_bboxes: bool = False,
                 interpolation: int = cv2.INTER_AREA):
        """
        Resize the input image to the given image's size.

        Args:
            image_size (int, tuple): A tuple containing the image's width and height in this order.
            resize_bboxes (bool): Specifies the use of resizing bounding boxes or not. Default: False.
            interpolation (int): The desired interpolation mode. Default: cv2.INTER_NEAREST.
        """
        self.image_size: Tuple[int, int] = image_size

        self.resize_bboxes: bool = resize_bboxes

        self.interpolation: int = interpolation

    def apply(self, image: np.ndarray, bboxes: Union[np.ndarray, None] = None):
        original_image_height, original_image_width = image.shape[:2]

        img_w, img_h = self.image_size

        image = cv2.resize(image, (img_w, img_h), interpolation=self.interpolation)

        if self.resize_bboxes and bboxes is not None:
            re_h, re_w = image.shape[:2]

            ratio_w = re_w / original_image_width

            ratio_h = re_h / original_image_height

            size_ = np.array([[ratio_w, ratio_h, ratio_w, ratio_h]])

            bboxes *= size_

        return image, bboxes


class ToTensor(AbstractTransformation):
    """
    Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W)
    in the range [0.0, 1.0].

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks.
    """

    def apply(self, image: np.ndarray, bboxes: Union[np.ndarray, None] = None):

        # Make sure the image has three dimensions.
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # Converting image from numpy shape: HxWxC to tensor shape: CxHxW
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).contiguous()

        if bboxes is not None and len(bboxes) != 0:
            bboxes = torch.from_numpy(bboxes.astype(np.float32))

        # Scaling image into range [0.0, 1.0]
        image = image / 255.0

        return image, bboxes


class Normalize(AbstractTransformation):

    def __init__(self, mean: List[float], std: List[float]):
        """
        Normalize a tensor image with the given mean and standard deviation.

        .. note::
            This transformation acts out of place, i.e., it does not mutate the input tensor.

        Args:
            mean (float, list): The list of means where its length represents the number of channels.
            std (float, list): The list of standard deviations where its length represents the number of channels.
        """
        self.std: np.ndarray = np.array(std)

        if (self.std == 0).any():
            raise ValueError("Standard deviation evaluated to zero after conversion. This leads to division by zero.")

        self.mean: np.ndarray = np.array(mean)

        # Checking whether the mean range lies within 0..1
        mean_in_range_zero_one = np.all((self.mean >= 0.0) & (self.mean <= 1.0))
        if not mean_in_range_zero_one:
            self.mean = self.mean / 255.0

        # Checking whether the std range lies within 0..1
        std_in_range_zero_one = np.all((self.std >= 0.0) & (self.std <= 1.0))
        if not std_in_range_zero_one:
            self.std = self.std / 255.0

    def to_tensor(self):
        """
        Converts the mean and standard deviation into tensor.

        Returns:
            The mean and standard deviation in tensor.

        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device="cpu")

        std = torch.as_tensor(self.std, dtype=torch.float32, device="cpu")

        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)

        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        return mean, std

    def apply(self, image: torch.Tensor, bboxes: Union[np.ndarray, None] = None) -> torch.Tensor:

        mean, std = self.to_tensor()

        image = (image - mean) / std

        return image, bboxes


class ConvertColor(AbstractTransformation):

    def __init__(self, current: Optional[str] = "RGB", transform: Optional[str] = "GRAY"):
        """
        Convert image from a current color to another given color

        Args:
            current (string, optional): The current image's color.
            transform (string, optional): The image's color to apply to.
        """
        self.current: str = current

        self.transform: str = transform

    def apply(self, image, bboxes: Union[np.ndarray, None] = None):

        if self.current == "RGB" and self.transform == "GRAY":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.current == "GRAY" and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise NotImplementedError("Those two colors '{0}':'{1}' are not currently supported!"
                                      .format(self.current, self.transform))
        return image, bboxes


# ToSobelGradient, ToMorphology and CropImage was taken from:
# https://github.com/eadst/CEIR/blob/master/preprocess/crop.py
class ToSobelGradient(AbstractTransformation):

    def __init__(self, thresh_value: int):
        """
        Perform an image's binarization with the help of sobel gradient.

        Args:
            thresh_value (int): The threshold value to binarize image.
        """
        self.thresh_value: int = thresh_value

    def apply(self, gray_image: np.ndarray):
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)

        # Sobel gradient
        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # thresh and blur
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        threshed_image = cv2.threshold(blurred, 0, 100, self.thresh_value)[1]

        return threshed_image


class ToMorphology(AbstractTransformation):
    """
    Perform the morphological transformation with the help of structuring element whose kernel size is set to
    (w / 40, h / 18) where w, h is the image's width and height respectively.
    """

    def apply(self, threshed_image: np.ndarray, erode_iterations: int, dilate_iterations: int):

        kernel_size = (int(threshed_image.shape[1] / 40), int(threshed_image.shape[0] / 18))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        morpho_image = cv2.morphologyEx(threshed_image, cv2.MORPH_CLOSE, kernel)

        morpho_image = cv2.erode(morpho_image, None, iterations=erode_iterations)

        morpho_image = cv2.dilate(morpho_image, None, iterations=dilate_iterations)

        return morpho_image


class CropImage(AbstractTransformation):

    def __init__(self, draw_contours: Optional[bool] = False):
        """
        Crop a given image based on the morphological image.

        Args:
            draw_contours (bool, optional): Specifies the use of drawing the image contours or not. Default: False.
        """
        self.draw_contours: bool = draw_contours

    def apply(self, morpho_image: np.ndarray, source_image: np.ndarray):
        contours, hierarchy = cv2.findContours(morpho_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        height = source_image.shape[0]
        weight = source_image.shape[1]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = max(min(Xs), 0)
        x2 = min(max(Xs), weight)
        y1 = max(min(Ys), 0)
        y2 = min(max(Ys), height)

        height = y2 - y1
        width = x2 - x1

        cropped_image = source_image[y1:y1 + height, x1:x1 + width]

        output = (cropped_image, x1, y1)

        if self.draw_contours:
            image_with_contours = cv2.drawContours(source_image, [box], -1, (0, 0, 255), 3)
            output = (cropped_image, image_with_contours, x1, y1)

        return output
