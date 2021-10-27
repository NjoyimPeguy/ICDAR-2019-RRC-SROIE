import os
import cv2
import sys
import math
import torch
import numpy as np

sys.path.append(os.getcwd())

# Apparently, using CV2 with dataloader where num_workers > 0
# you will encounter a deadlock. This is because of CV2 using multithreading.
# The solution is to set the number of threads to zero.
# For further info, check this out:
# https://stackoverflow.com/questions/54013846/pytorch-dataloader-stucked-if-using-opencv-resize-method
cv2.setNumThreads(0)

import os.path as osp

from typing import Tuple, List
from vizer.draw import draw_boxes
from text_localization.ctpn.utils.dset import read_image, parse_annotations

random_state = np.random.RandomState()


def compute_intersection_numpy(gt_boxes: np.ndarray, cropped_boxes: np.ndarray) -> np.ndarray:
    """
        Compute the intersection between each and every two set of bounding boxes.

        Args:
            gt_boxes: The set of ground truth boxes. Shape: [N, 4].
            cropped_boxes: The cropped boxes. Shape: [1, 4]

        Returns:
            The intersection between ground truth boxes and the cropped box.

    """
    
    overlaps_top_left = np.maximum(gt_boxes[:, :2], cropped_boxes[:, 2:])
    
    overlap_bottom_right = np.minimum(gt_boxes[:, 2:], cropped_boxes[:, 2:])
    
    diff = overlap_bottom_right - overlaps_top_left
    
    clip_ = np.maximum(0.0, diff)
    
    intersection = clip_[:, 0] * clip_[:, 1]
    
    return intersection


def jaccard_index_numpy(gt_boxes: np.ndarray, cropped_boxes: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
        Compute the IoU between each and every two sets of bounding boxes.

        Args:
            gt_boxes: The ground truth coordinates. Shape: [N, 4].
            cropped_boxes: The cropped box coordinates. Shape: [1, 4].
            eps: a small number to avoid 0 as denominator.

        Returns:
            The Jaccard index/overlap between ground truth boxes and the cropped box.

    """
    
    # Computing the intersection between two sets of bounding boxes.
    intersection = compute_intersection_numpy(gt_boxes=gt_boxes, cropped_boxes=cropped_boxes)
    
    # Computing the area of each bounding box in the set of ground truth boxes.
    gt_box_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    
    # Computing the area of each bounding box in the set of cropped boxes.
    cropped_box_areas = (cropped_boxes[:, 2] - cropped_boxes[:, 0]) * (cropped_boxes[:, 3] - cropped_boxes[:, 1])
    
    # Computing the union area between ground truth and cropped boxes.
    union_area = gt_box_areas + cropped_box_areas - intersection
    
    # Computing the intersection over union between the two sets of bounding boxes.
    IoU = intersection / (union_area + eps)
    
    return IoU


class Compose(object):
    
    def __init__(self, transforms):
        self.transformations = transforms
    
    def __call__(self, img, bboxes=None):
        for transformation in self.transformations:
            img, bboxes = transformation(img, bboxes)
        return img, bboxes


class Resize(object):
    
    def __init__(self, image_size: Tuple[int, int], resize_bboxes: bool = False, interpolation=cv2.INTER_NEAREST_EXACT):
        self.image_size = image_size
        self.resize_bboxes = resize_bboxes
        self.interpolation = interpolation
    
    def __call__(self, image: np.ndarray, bboxes=None):
        original_image_height, original_image_width = image.shape[:2]
        
        image = cv2.resize(image, self.image_size, interpolation=self.interpolation)
        
        if self.resize_bboxes and bboxes is not None:
            re_h, re_w = image.shape[:2]
            ratio_w = re_w / original_image_width
            ratio_h = re_h / original_image_height
            size_ = np.array([[ratio_w, ratio_h, ratio_w, ratio_h]])
            bboxes *= size_
        
        return image, bboxes


class ToTensor(object):
    def __call__(self, image: np.ndarray, bboxes=None):
        
        # Checking whether the image range lies within 0..1
        image_in_range_zero_one = np.all((image >= 0.0) & (image <= 1.0))
        
        # Converting image from numpy shape: HxWxC to tensor shape: CxHxW
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).contiguous()
        
        if bboxes is not None and len(bboxes) != 0:
            bboxes = torch.from_numpy(bboxes.astype(np.float32))
        
        if not image_in_range_zero_one:
            image = image / 255.0
        
        return image, bboxes


class Normalize(object):
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean)
        self.std = np.array(std)
        
        # Checking whether the mean range lies within 0..1
        mean_in_range_zero_one = np.all((self.mean >= 0.0) & (self.mean <= 1.0))
        if not mean_in_range_zero_one:
            self.mean = self.mean / 255.0
        
        # Checking whether the std range lies within 0..1
        std_in_range_zero_one = np.all((self.std >= 0.0) & (self.std <= 1.0))
        if not std_in_range_zero_one:
            self.std = self.std / 255.0
    
    def __call__(self, image: torch.Tensor, boxes=None) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {},'
                             'leading to division by zero.'.format(image.dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        
        image = (image - mean) / std
        
        return image, boxes


class ConvertFromInts(object):
    def __call__(self, image, bboxes=None):
        return image.astype(np.float32), bboxes


class ConvertColor(object):
    def __init__(self, current, transform):
        self.current = current
        self.transform = transform
    
    def __call__(self, image, bboxes=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif self.current == "RGB" and self.transform == "GRAY":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.current == 'GRAY' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise NotImplementedError
        return image, bboxes


class RandomHorizontalFlip(object):
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image, bboxes):
        if random_state.rand() > self.p:
            return image, bboxes
        width = image.shape[1]
        image = image[:, ::-1, :]
        bboxes = bboxes.copy()
        if bboxes.size > 0:
            bboxes[:, 0::2] = width - bboxes[:, 2::-2]
        return image, bboxes


class RandomZoomOut(object):
    def __init__(self, bg_color: List[float], p: float = 0.5):
        self.p = p
        self.bg_color = np.array(bg_color)
        in_range_zero_one = np.all((self.bg_color >= 0.0) & (self.bg_color <= 1.0))
        if in_range_zero_one:
            self.bg_color = self.bg_color * 255.0
    
    def __call__(self, image, bboxes):
        if random_state.rand() > self.p:
            return image, bboxes
        
        height, width, depth = image.shape
        ratio = random_state.uniform(1, 2)
        left = random_state.uniform(0, width * ratio - width)
        top = random_state.uniform(0, height * ratio - height)
        
        expand_image = np.ones((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.bg_color
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        
        image = expand_image
        
        bboxes = bboxes.copy()
        bboxes[:, :2] += (int(left), int(top))
        bboxes[:, 2:] += (int(left), int(top))
        
        return image, bboxes


class RandomZoomIn(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
    """
    
    def __init__(self, p: float = 0.5, sample_options: tuple = (
            # randomly sample a patch
            None,
            # sample a patch s.t. MIN jaccard w/ obj in 0.1, 0.3, 0.5, 0.7, 0.9
            0.1,
            0.3,
            0.5,
            0.7,
            0.9)):
        
        self.p = p
        self.sample_options = sample_options
    
    def __call__(self, image: np.ndarray, bboxes=None):
        
        # guard against no boxes
        if bboxes is not None and len(bboxes) == 0:
            return image, bboxes
        
        original_height, original_width = image.shape[:2]
        
        while True:
            # Randomly choose a mode.
            mode = self.sample_options[random_state.randint(0, len(self.sample_options))]
            
            # No cropping, i.e., use the entire original image.
            if random_state.rand() > self.p:
                return image, bboxes
            
            min_IoU = mode
            if min_IoU is None:
                min_IoU = float("-inf")
            
            max_trials = 50
            for _ in range(max_trials):
                min_scale = 0.3
                new_w = random_state.uniform(min_scale * original_width, original_width)
                new_h = random_state.uniform(min_scale * original_height, original_height)
                
                # Aspect ratio has to be in [0.5, 2].
                aspect_ratio = new_h / new_w
                if not (0.5 < aspect_ratio < 2):
                    continue
                
                # Crop coordinates (origin at top-left of image).
                left = int(random_state.uniform(original_width - new_w))
                right = int(left + new_w)
                top = int(random_state.uniform(original_height - new_h))
                bottom = int(top + new_h)
                
                # Converting to x1, y1, x2, y2.
                cropped_box = np.array([left, top, right, bottom], dtype=bboxes.dtype)  # (4)
                
                # Cut the crop from the image.
                new_image = image[top:bottom, left:right, :]  # (new_h, new_w, 3)
                
                # Calculate IoU (jaccard overlap) b/t the cropped and gt boxes.
                IoU = jaccard_index_numpy(bboxes, np.expand_dims(cropped_box, axis=0))
                
                # We want to accept a crop as valid if at least one box meets the minimum Jaccard overlap.
                # Otherwise, try again.
                if max(IoU) < min_IoU:
                    continue
                
                # keep overlap with gt box if center in sampled patch
                bbox_centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
                
                # Find bounding boxes whose centers are in the crop
                m1 = (bbox_centers[:, 0] > left) * (bbox_centers[:, 1] > top)
                m2 = (bbox_centers[:, 0] < right) * (bbox_centers[:, 1] < bottom)
                mask = m1 * m2
                
                # # If not a single bounding box has its center in the crop, try again.
                if not mask.any():
                    continue
                
                # Discard bounding boxes that don't meet this criterion
                # i.e., take only matching ground truth boxes.
                new_boxes = bboxes[mask, :].copy()
                
                # Calculate bounding boxes' new coordinates in the crop
                top_left = cropped_box[:2]
                right_bottom = cropped_box[2:]
                
                # Should we use the top left corner or the crop's.
                new_boxes[:, :2] = np.maximum(new_boxes[:, :2], top_left)
                
                # Adjusting the crop (by substracting top left's crop).
                new_boxes[:, :2] -= top_left
                
                # Should we use the bottom right corner or the crop's.
                new_boxes[:, 2:] = np.minimum(new_boxes[:, 2:], right_bottom)
                
                # Adjusting the crop (by substracting top left's crop).
                new_boxes[:, 2:] -= top_left
                
                return new_image, new_boxes


# ToSobelGradient, ToMorphology and CropImage was taken from https://github.com/eadst/CEIR/blob/master/preprocess/crop.py
class ToSobelGradient(object):
    def __init__(self, thresh_value):
        self.thresh_value = thresh_value
    
    def __call__(self, gray_image):
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
        # Sobel gradient
        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        # thresh and blur
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        threshed_image = cv2.threshold(blurred, 0, 255, self.thresh_value)[1]
        return threshed_image


class ToMorphology(object):
    def __call__(self, threshed_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (int(threshed_image.shape[1] / 40), int(threshed_image.shape[0] / 18)))
        morpho_image = cv2.morphologyEx(threshed_image, cv2.MORPH_CLOSE, kernel)
        morpho_image = cv2.erode(morpho_image, None, iterations=1)
        morpho_image = cv2.dilate(morpho_image, None, iterations=1)
        return morpho_image


class CropImage(object):
    
    def __init__(self, draw_contours: bool = False):
        self.draw_contours = draw_contours
    
    def __call__(self, morpho_image, source_image):
        contours, hierarchy = cv2.findContours(morpho_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
            image_with_contours = cv2.drawContours(source_image.copy(), [box], -1, (0, 0, 255), 3)
            output = (cropped_image, image_with_contours, x1, y1)
        return output


class SplitBoxes(object):
    
    def __init__(self, anchor_scale: int):
        """
        Split the ground truth boxes into a sequence of fine-scale text proposals,
        where each proposal generally represents a small part of a text line, e.g., a text piece with 16-pixel width.
        
        Args:
            anchor_scale: The scale of each anchor box.
            
        """
        
        self.anchor_scale = anchor_scale
    
    def __call__(self, image: np.ndarray, gt_bboxes: np.ndarray):
        new_gt_bboxes = []
        for bbox in gt_bboxes:
            xmin, ymin, xmax, ymax = bbox
            bbox_ids = np.arange(int(math.floor(1. * xmin / self.anchor_scale)),
                                 int(math.ceil(1. * xmax / self.anchor_scale)))
            new_bboxes = np.zeros(shape=(len(bbox_ids), 4))
            new_bboxes[:, 0] = bbox_ids * self.anchor_scale
            new_bboxes[:, 1] = ymin
            new_bboxes[:, 2] = (bbox_ids + 1) * self.anchor_scale
            new_bboxes[:, 3] = ymax
            new_gt_bboxes.append(new_bboxes)
        
        new_gt_bboxes = np.concatenate(new_gt_bboxes, axis=0)
        
        return image, new_gt_bboxes


if __name__ == "__main__":
    path = osp.normpath("text_localization/ctpn/data/preprocessing/transformations/demo/images/X00016469612.jpg")
    filename = osp.basename(path)
    original_image = np.array(read_image(path))
    
    annotation_path = osp.normpath(
        "text_localization/ctpn/data/preprocessing/transformations/demo/annotations/X00016469612.txt")
    bboxes = parse_annotations(annotation_path)
    
    image, splitted_bboxes = SplitBoxes(anchor_scale=16.)(original_image, bboxes)
    drawn_image = draw_boxes(image=image, boxes=splitted_bboxes)
    cv2.namedWindow("Image with splitted bounding boxes", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Image with splitted bounding boxes", drawn_image)
    cv2.resizeWindow("Image with splitted bounding boxes", 1000, 950)
    
    zoomedImage, bboxes = RandomHorizontalFlip()(original_image, bboxes)
    zoomedImage, bboxes = RandomZoomOut(bg_color=[0.92088137, 0.92047861, 0.92000766])(zoomedImage, bboxes)
    zoomedImage, bboxes = RandomZoomIn()(zoomedImage, bboxes)
    augmented_image = draw_boxes(image=zoomedImage, boxes=bboxes)
    cv2.namedWindow("Augmented image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Augmented image", augmented_image)
    cv2.resizeWindow("Augmented image", 1000, 950)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
