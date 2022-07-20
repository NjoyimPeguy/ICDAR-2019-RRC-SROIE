import os
import cv2
import math
import torch
import numpy as np
import os.path as osp

from typing import Optional, Tuple
from torch.utils.data import Dataset
from functional.utils.box import order_point_clockwise, to_xy_min_max


def compute_mean_and_std(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and std of a given dataset.

    Args:
        dataset (Dataset): The given dataset.

    Returns:
        A tuple containing the mean and std of the given dataset.

    """
    n_images = len(dataset)

    channels_sum = np.array([0.0, 0.0, 0.0])

    channels_squared_sum = np.array([0.0, 0.0, 0.0])

    for idx in range(n_images):

        image, _ = dataset[idx]

        image = image / 255.0

        C = image.shape[2]

        image = image.reshape(C, -1)

        channels_sum += np.mean(image, axis=1)

        channels_squared_sum += np.mean(image ** 2, axis=1)

    mean = channels_sum / n_images

    # std = sqrt(E[X^2] - (E[X])^2)
    std = np.sqrt((channels_squared_sum / n_images) - np.power(mean, 2))

    return mean, std


def make_directories(dir_names):
    """
    Create directories if they do not exist.

    Args:
        dir_names: The directory names.

    """
    if not osp.exists(dir_names):
        os.makedirs(dir_names)


def compute_class_weights(classes: np.ndarray, mu: float = 0.15):
    """
    Compute the class weights for unbalanced datasets.

    Args:
        classes: Array of the original classes.
        mu: A smooth value.

    Returns:
        Array with class_weights[i] the weight for i-th class.

    """
    class_occurrences = {}

    unique_classes = np.unique(classes)

    for klass in unique_classes:
        class_occurrences[klass] = np.sum(classes == klass)

    total_number_of_classes = np.sum(list(class_occurrences.values()))

    class_weights = np.zeros_like(unique_classes, dtype=np.float32)

    for klass, occurrences in class_occurrences.items():
        class_weight = math.log((mu * total_number_of_classes) / float(occurrences))

        class_weights[klass] = class_weight if class_weight > 1.0 else 1.0

    return class_weights


def read_image(path_to_image: str):
    """
    Read an input image from a location and convert it to RGB.

    Args:
        path_to_image (string): The path to the image,

    Returns:
        A numpy array which represents the image.

    """

    bgr_image = cv2.imread(path_to_image)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    return rgb_image


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


def parse_annotations(annotation_path: str, max_parts: int = 8):
    """
    Parse the annotation file to extract the bounding box coordinates.

    Args:
        annotation_path: The path to the annotation file.
        max_parts: The number of parts the bounding box coordinates have.

    Returns:
        A numpy array containing the bounding box coordinates in (xmin, ymin, xmax, ymax) format.

    """
    bboxes = []
    with open(annotation_path, encoding='utf-8', mode='r', errors='ignore') as file:
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
