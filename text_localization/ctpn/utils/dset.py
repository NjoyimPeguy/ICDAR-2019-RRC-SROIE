import os
import cv2
import numpy as np
import os.path as osp

from PIL import Image
from PIL.ImageStat import Stat
from torchvision.transforms import ToPILImage
from text_localization.ctpn.utils.box import order_point_clockwise, to_xy_min_max


# Taken from https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/13
class DatasetStats(Stat):
    def __add__(self, other):
        return DatasetStats(list(np.add(self.h, other.h)))


def compute_mean_and_std(dataset):
    """
    Compute the mean and std of a given dataset.
    
    Args:
        dataset: The given dataset.

    Returns:
        A tuple containing the mean and std of the given dataset.
        
    """
    statistics = None
    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        if statistics is None:
            statistics = DatasetStats(ToPILImage()(image))
        else:
            statistics += DatasetStats(ToPILImage()(image))
    mean = np.array(statistics.mean)
    std = np.array(statistics.stddev)
    output = (mean, std)
    return output


def make_directories(dir_names):
    """
    Create directories if they do not exist.
    
    Args:
        dir_names: The directory names.
        
    """
    if not osp.exists(dir_names):
        os.makedirs(dir_names)


def read_image(path_to_image):
    """
    Read an input image from a location and convert it to RGB.
    
    Args:
        path_to_image: The path to the image,

    Returns:
        A numpy array which represents the image.
        
    """
    
    image = np.array(Image.open(path_to_image).convert("RGB"))
    
    return image


def read_image_ids(path_to_image_ids, filename):
    """
    Read the image name (without the extension) or IDs from a given location.
    
    Args:
        path_to_image_ids: The path to the given image IDs.
        filename: The name of the file.

    Returns:
        A numpy array containing the image IDs or name (without the extension).
        
    """
    image_sets_file = osp.join(path_to_image_ids, filename)
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return np.array(ids)


def parse_annotations(annotation_path, max_parts: int = 8):
    """
    Parse the annotation file to extract the bounding box coordinates.
    
    Args:
        annotation_path: The path to the annotation file.
        max_parts: The number of parts the bounding box coordinates have.

    Returns:
        A numpy array containing the bounding box coordinates.
        
    """
    bboxes = []
    with open(annotation_path, encoding='utf-8', mode='r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            if len(parts) > 1:
                bbox = order_point_clockwise(np.array(list(map(np.float32, parts[:max_parts]))).reshape((4, 2)))
                if cv2.arcLength(bbox, True) > 0:
                    bbox = np.array(to_xy_min_max(bbox.flatten()))
                    bboxes.append(bbox)
    bboxes = np.array(bboxes, dtype=np.float32)
    return bboxes


if __name__ == "__main__":
    pts = np.array([
        [72, 25],
        [326, 25],
        [326, 64],
        [72, 64]])
    
    ordered_points = order_point_clockwise(pts)
    print(ordered_points.astype("int"))
