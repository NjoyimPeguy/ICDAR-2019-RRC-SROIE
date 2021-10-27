import os
import csv
import cv2
import regex
import random
import numpy as np
import os.path as osp

from fuzzywuzzy import fuzz
from scripts.datasets.dataset_roots import SROIE_ROOT
from keyword_information_extraction.utils import get_basename
from keyword_information_extraction.utils.sroie_utils import read_json_file, reorder_json_file
from keyword_information_extraction.data.datasets.sroie2019.variables import SROIE_DATE_PATTERN, SROIE_TOTAL_PATTERN

# Importing useful function from task one.
from text_localization.ctpn.utils.box import to_xy_min_max, order_point_clockwise


class TextBox(object):
    def __init__(self, coordinates: list, max_parts: int):
        """
        An abstract class representing a bounding box.
        
        Args:
            coordinates: The bounding box coordinates.
            max_parts: The maximum number of parts the bounding box coordinates have.
            
        """
        bbox = np.array(list(map(np.float32, coordinates[:max_parts])))
        self.x, self.y = to_xy_min_max(bbox)[:2]
        self.text = ",".join(coordinates[:max_parts + 1])  # +1 because we also want the text.
    
    def __str__(self):
        return self.text


def is_number(inputString):
    """
    Check whether the given input string is a number or not.
    
    Args:
        inputString: The input string to verify.

    Returns:
        True if the given input string is a number. Otherwise, False.
        
    """
    return all(char.isdigit() for char in inputString)


def parse_annotation(annotation_path):
    """
    Parse the annotations given a path.
    
    Args:
        annotation_path: The path to the annotation.

    Returns:
        A list of dict where each dictionary contains the filename, bounding box coordinates and the text.
        
    """
    ocr_data = []
    max_parts = 8
    filename = get_basename(annotation_path)
    with open(annotation_path, encoding='utf-8', mode='r', errors='ignore') as file:
        lines = file.readlines()
        text_boxes = []
        for line in lines:
            parts = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',', maxsplit=max_parts)
            if len(parts) > 1:
                text_boxes.append(TextBox(parts, max_parts=max_parts))
        # Firstly, sort from left to right
        lines = sorted(text_boxes, key=lambda box: box.x, reverse=False)
        # Secondly, sort from top to bottom
        lines = sorted(lines, key=lambda box: box.y, reverse=False)
        for line in lines:
            data = {}
            parts = str(line).strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            if len(parts) > 1:
                ordered_bbox = order_point_clockwise(np.array(list(map(np.float32, parts[:max_parts]))).reshape((4, 2)))
                if cv2.arcLength(ordered_bbox, True) > 0:
                    text = ",".join(parts[max_parts:])
                    data["filename"] = filename
                    data["text"] = text
                    ocr_data.append(data)
    return ocr_data


def assign_label_to_line(line2Match: str, entity: dict, thresh_ratio: float = 0.6):
    """
    Assign a label to a given string text line.
    
    Args:
        line2Match: The string line to match.
        entity: The OCR data for a given txt file.
        thresh_ratio: A threshold (i.e., a float number) which defines a good or bad assigned label.

    Returns:
        A string that is the assigned label given a string line.
        
    """
    entity = reorder_json_file(old_entity=entity, old_ordered_keys=["company", "address", "date", "total"])
    
    for gt_label in entity.keys():
        entityLine = entity[gt_label]
        if gt_label in ("company", "address") and not is_number(line2Match.replace(".", "")):
            # token_set_ratio is used here because we have two strings of widely different lengths.
            ratio = fuzz.token_set_ratio(line2Match, entityLine) / 100.0
            if ratio > thresh_ratio:
                return gt_label
        else:
            matched_textline = None
            if gt_label == "date":
                line = line2Match.replace("[", "").replace("]", "").replace(":", "")
                matched_textline = regex.search(SROIE_DATE_PATTERN, line.strip())
            elif gt_label == "total":
                matched_textline = regex.search(SROIE_TOTAL_PATTERN, line2Match.strip())
            if matched_textline is not None:
                if entityLine.strip() in matched_textline.group():
                    return gt_label
    return "none"


def assign_labels(ocr_data: list, entity: dict, class_dict: dict):
    """
    Given a txt file containing the OCR data, this will assign a label to each line.
    
    Args:
        ocr_data: A list of dictionaries for which each line is not assigned to a given label.
        entity: The corresponding json file to the txt file containing the OCR data.
        class_dict: The class dictionaries.

    Returns:
        A list of dictionaries where each line is assigned to a label.
        
    """
    
    already_labeled = {
        "company": False,
        "address": False,
        "date": False,
        "total": False
    }
    
    # For each data line, which is a dict containing the filename,
    # the four bounding box coordinates and the text in a receipt
    for i, data_line in enumerate(ocr_data):
        text_line = data_line["text"]
        label = assign_label_to_line(text_line, entity)  # Assign a label to this text line
        already_labeled[label] = True
        
        if label == "company" and already_labeled["address"]:
            label = "address"
        
        if label == "address" and already_labeled["total"]:
            label = "none"
        
        data_line["label"] = label
        data_line["class"] = class_dict[label]
    return ocr_data


def generate_csv_for_training(save_folder, class_dict: dict, validation_args: dict):
    """
    Generate the csv files for the training phase.
    
    Args:
        save_folder: The path where the csv files will be.
        class_dict: The dictionary of classes
        validation_args: The dictionary of validation arguments.
        
    """
    
    print("\nGenerating the csv file for training...\n")
    path_to_task1_training_set = osp.join(SROIE_ROOT, "0325updated.task1train(626p)")
    if not osp.exists(path_to_task1_training_set):
        raise ValueError("It seems you have completely forgotten to download the main dataset!")
    task1_training_files = np.array(
        list(sorted(os.scandir(path=path_to_task1_training_set), key=lambda file: file.name))
    )
    task1_training_files = [get_basename(file.path) for file in task1_training_files if file.name.endswith("txt")]
    
    path_to_task2_training_set = osp.join(SROIE_ROOT, "0325updated.task2train(626p)")
    task2_training_files = np.array(
        list(sorted(os.scandir(path=path_to_task2_training_set), key=lambda file: file.name))
    )
    task2_training_files = [get_basename(file.path) for file in task2_training_files if file.name.endswith("txt")]
    
    train_filenames = sorted(list(set(task2_training_files).intersection(task1_training_files)))
    
    csv_columns = ["filename", "text", "label", "class"]
    
    train_ratio = 1.0 - validation_args["ratio"]
    random_state = np.random.RandomState()
    val_data = validation_args["val_folder"]
    val_folder = osp.join(SROIE_ROOT, val_data["data-dir"], val_data["split"])
    if not osp.isdir(val_folder):
        os.makedirs(val_folder)
    
    # all_texts = []
    # text_max_length = 0
    
    for i, train_filename in enumerate(train_filenames):
        txt_file = train_filename + ".txt"
        annotation_path = osp.join(path_to_task1_training_set, txt_file)
        json_path = osp.join(path_to_task2_training_set, txt_file)
        entity = read_json_file(json_path)
        if entity is not None:  # If this json file does not contains non empty text category.
            ocr_data = parse_annotation(annotation_path)
            lines_labeled = assign_labels(ocr_data, entity, class_dict)  # the assigned labels for this file.
            folder = save_folder
            
            p = random_state.rand()
            if p > train_ratio:
                folder = val_folder
            
            csv_file = osp.join(folder, train_filename + ".csv")
            try:
                with open(csv_file, 'w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=",")
                    writer.writeheader()
                    for data_line in lines_labeled:
                        writer.writerow(data_line)
                        # text_line = data_line["text"].strip()
                        # current_text_max_length = len(text_line)
                        # text_max_length = max(text_max_length, current_text_max_length)
                        # all_texts.append(text_line)
            except IOError:
                print("I/O error")
    
    # print("Generating the vocabulary for training...\n")
    # parent_dir = osp.abspath(osp.join(save_folder, os.pardir))
    # txt_file = osp.join(parent_dir, "train-vocabulary.txt")
    # try:
    #     with open(txt_file, 'w') as f:
    #         sentence = " ".join([text.upper() for text in all_texts])
    #         vocabulary = "".join(sorted(set(sentence)))
    #         f.write(vocabulary)
    # except IOError:
    #     print("I/O error")
    
    # txt_file = osp.join(parent_dir, "text-max-length.txt")
    # try:
    #     with open(file=txt_file, mode='w') as f:
    #         f.write(str(text_max_length))
    # except IOError:
    #     print("I/O error")


def generate_csv_for_evaluation(test_folder):
    """
    Generate the csv files for the test/evaluation phase.
    
    Args:
        test_folder: The path where the csv files will be.
        
    """
    
    print("\nGenerating the csv file for the evaluation...\n")
    path_to_task3_test_set = osp.join(SROIE_ROOT, "task3-test 347p) -", "task3-test（347p)")
    if not osp.isdir(path_to_task3_test_set):
        raise ValueError("it seems you did not download the dataset!")
    
    task3_jpg_files = np.array(list(sorted(os.scandir(path=path_to_task3_test_set), key=lambda file: file.name)))
    task3_filenames = [get_basename(file.path) for file in task3_jpg_files if file.name.endswith("jpg")]
    
    path_to_task2_annotations = osp.join(SROIE_ROOT, "text.task1_2-test（361p)")
    task2_txt_files = np.array(list(sorted(os.scandir(path=path_to_task2_annotations), key=lambda file: file.name)))
    task2_filenames = [get_basename(file.path) for file in task2_txt_files if file.name.endswith("txt")]
    
    test_filenames = sorted(list(set(task3_filenames).intersection(task2_filenames)))
    
    csv_columns = ["filename", "text"]
    for test_filename in test_filenames:
        txt_file = test_filename + ".txt"
        annotation_path = osp.join(path_to_task2_annotations, txt_file)
        ocr_data = parse_annotation(annotation_path)
        csv_file = osp.join(test_folder, test_filename + ".csv")
        try:
            with open(csv_file, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=",")
                writer.writeheader()
                for data_line in ocr_data:
                    writer.writerow(data_line)
        except IOError:
            print("I/O error")


if __name__ == '__main__':
    
    json_path = osp.join(SROIE_ROOT, "0325updated.task2train(626p)", "X51005442333.txt")
    json_file = read_json_file(json_path)
    print(json_file)
    print("\n")
    
    annotation_path = osp.join(SROIE_ROOT, "0325updated.task1train(626p)", "X51005442333.txt")
    ocr_data = parse_annotation(annotation_path)
    print(ocr_data)
    print("\n")
    
    text_line = ocr_data[24]["text"]
    label = assign_label_to_line(text_line, json_file)
    print("line2Match:", text_line)
    print("label: ", label)
    print("\n")
    
    print("Fuzzy token_set_ratio test")
    line2Match = "$6.90"
    lineOption = "NETT TOTAL:$6.90"
    ratio = fuzz.token_set_ratio(line2Match, lineOption) / 100.0
    print(ratio)
    print("\n")
    
    str2Matches = ["- 75.00", "RM85.00", "$8.55", "RM 65.00", "85.00SR"]
    for str2Match in str2Matches:
        pattern = r"^[^\+\-]+(\d+\.(\d{2}|\d]))(?![\s+0-9A-Z])"
        matched = regex.search(pattern, str2Match.strip())
        if matched is not None:
            print("Could matched {0}".format(matched.group()))
        else:
            print("Could not matched {0}".format(str2Match))
    
    str2Match = "DATE: 2018-03-23"
    matched = regex.search(SROIE_DATE_PATTERN, str2Match.strip())
    if matched is not None:
        print("Could matched {0}".format(matched.group()))
    else:
        print("Could not matched {0}".format(str2Match))
