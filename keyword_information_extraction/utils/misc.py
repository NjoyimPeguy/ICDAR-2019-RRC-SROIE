import json
import numpy as np
import os.path as osp

from typing import List

from functools import cmp_to_key

from scripts.datasets.dataset_roots import SROIE_ROOT

from functional.utils.dataset import read_image
from functional.utils.box import to_xy_min_max, order_point_clockwise


class TextBox(object):
    def __init__(self, bbox_transcript: list, max_parts: int):
        """
        An abstract class representing a bounding box with its transcription part.

        Args:
            bbox_transcript (list): The bounding box coordinates and its transcription.
            max_parts (int): The maximum number of parts the bounding box coordinates have.

        """

        bbox = np.array(list(map(np.float32, bbox_transcript[:max_parts])))

        bbox = order_point_clockwise(bbox.reshape((4, 2))).flatten()

        self.x, self.y = np.array(to_xy_min_max(bbox))[:2]

        self.text = bbox_transcript[max_parts:][0]

        self.text = bbox_transcript[max_parts:][0]

    def __str__(self):
        return self.text


def get_basename(path_to_file: str):
    """
    Get the base name in specified path.

    Args:
        path_to_file (string): The file's path.

    Returns:
        The base name of this file.
    """
    return osp.splitext(osp.basename(path_to_file))[0]


def is_number(inputString: str):
    """
    Check whether the given input string is a number or not.

    Args:
        inputString (string): The input string to verify.

    Returns:
        True if the given input string is a number. Otherwise, False.

    """
    return all(char.isdigit() for char in inputString)


def check_denominator_consistency(denominator: np.ndarray):
    """
    Replace all zeros with ones to the data that is supposed to be divided by a numerator.

    Args:
        denominator (numpy array): The data to be divided by a numerator.

    Returns:
        A new data where all zeros are replaced by ones.
    """
    mask = denominator == 0.0
    denominator[mask] = 1.0
    return denominator


def read_json_file(json_file: str, labels_classes: dict):
    """
    Read a given JSON path_to_file.
    
    Args:
        json_file (string): The JSON path_to_file to read to.
        labels_classes (dict): The labels (keys) & classes (values) dictionary.

    Returns:
        A new JSON path_to_file containing the entities (keys) and tuples (values)
            where each tuple contains a class and the entity text.
            It is worth mentioning that a new entity 'none' is added.
        
    """

    with open(json_file, "r") as file:
        entities = json.load(file)

    # Creating a new entities with one additional key, 'none'.
    new_entities = {k: (v, "") for k, v in labels_classes.items()}

    # Looping through each entity
    for entity, (klass, _) in new_entities.items():

        entity_text = entities.get(entity)

        if entity_text is not None:
            # Storing this entity alongside with its class.
            new_entities.update({entity: (klass, entity_text)})

    return new_entities


def get_bbox_precedence(firstBBox: TextBox, secondBBox: TextBox, tolerance_factor: int = 14):
    """
    Compute the text boxes precedence.

    Args:
        firstBBox (TextBox): The first text box.
        secondBBox (TextBox): The second text box.
        tolerance_factor (int): A tolerance factor.

    Returns:
        The bounding box precedence.
    """

    # This condition is needed for the cases where the right boxes are higher than the left boxes.
    if abs(firstBBox.y - secondBBox.y) <= tolerance_factor:
        return firstBBox.x - secondBBox.x

    return firstBBox.y - secondBBox.y


def parse_annotation(annotation_path: str, max_parts: int = 8):
    """
    Parse the annotations given a path.

    Args:
        annotation_path (string): The path to the annotation.
        max_parts (int): The maximum parts of a text line when split.

    Returns:
        A list of dict where each dictionary contains the filename and the text.

    """

    ocr_data = []

    filename = get_basename(annotation_path)

    with open(annotation_path, encoding='utf-8', mode='r', errors='ignore') as file:

        lines = file.readlines()

        text_boxes = []

        for line in lines:
            parts = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',', maxsplit=max_parts)
            if len(parts) > 1:
                text_boxes.append(TextBox(bbox_transcript=parts, max_parts=max_parts))

        text_boxes = sorted(text_boxes, key=cmp_to_key(get_bbox_precedence), reverse=False)

        for text in text_boxes:
            ocr_data.append({"filename": filename, "text": str(text)})

    return ocr_data


def multilabel_confusion_matrix(inputs: List[int], targets: List[int], n_classes: int):
    r"""

    Compute class-wise multilabel confusion matrix to evaluate the accuracy of a classification,
    and output confusion matrices for each class or sample. In multilabel confusion matrix :math:`MCM`,
    the count of true negatives is :math:`MCM_{[:,0,0]}`, false negatives is :math:`MCM_{[:,1,0]}`,
    true positives is :math:`MCM_{[:,1,1]}` and false positives is :math:`MCM_{[:,0,1]}`.

    Args:
        inputs (int, list): The estimated/predicted targets returned by a classifier/model.
        targets (int, list): The ground truth (correct) target values.
        n_classes (int): The number of classes

    Returns:
        multi_confusion : tensor shape (:, 2, 2). A confusion matrix corresponding to each output in the input.
            The results are returned in sorted order.

    """

    inputs = np.array(inputs)

    targets = np.array(targets)

    # We need to filter out ignored classes and put it into 1D-tensor as we are going to use torch.bincount
    mask = (targets >= 0) & (targets < n_classes)
    new_targets = targets[mask]
    predictions = inputs[mask]

    # Now as labels are from 0 to C - 1, we can use bincount which
    # counts the number of occurrences of each value in array of non-negative ints.
    tp = predictions == new_targets
    tp_bins = new_targets[tp]

    true_sum = pred_sum = tp_sum = np.zeros(shape=(n_classes,))

    if len(tp_bins) != 0:
        tp_sum = np.bincount(tp_bins, minlength=n_classes)

    if len(predictions) != 0:
        pred_sum = np.bincount(predictions, minlength=n_classes)

    if len(new_targets) != 0:
        true_sum = np.bincount(new_targets, minlength=n_classes)

    # Retain only selected labels
    labels = np.unique(np.concatenate([predictions, new_targets], axis=0))
    indices = np.searchsorted(labels, labels[:len(labels)])

    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]
    pred_sum = pred_sum[indices]

    tp = tp_sum
    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tn = len(new_targets) - tp - fp - fn

    MCM = np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)

    return MCM
