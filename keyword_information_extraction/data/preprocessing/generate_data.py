import os
import sys

sys.path.append(os.getcwd())

import csv
import regex
import numpy as np
import os.path as osp

from typing import Optional

from functional.utils.dataset import read_image

from scripts.datasets.dataset_roots import SROIE_ROOT

from keyword_information_extraction.utils.misc import get_basename, read_json_file, parse_annotation


def compute_entity_classes(entities: dict, text_space: str, n_errors: Optional[int] = 11):
    """
    Compute the class for each entity in the text-space.

    Args:
        entities (dict): The corresponding json path_to_file to the txt path_to_file containing the OCR data.
        text_space (string): A string representing the all OCR data.
        n_errors (int, optional): The total number of errors allowed.

    Returns:
        An array of class indices where the arr[i] corresponds to the class of a text line in a given OCR data.

    """
    none_class = entities["none"][0]

    text_lines = text_space.split("\n")

    entity_classes = np.full(shape=(len(text_lines, )), fill_value=none_class, dtype=np.int64)

    # For each entity
    for entity in entities.keys():

        # Skip the 'none' entity.
        if entity != "none":

            klass, entity_text = entities[entity]

            # Remove trailing spaces.
            entity_text = entity_text.strip()

            # If this entity has a text
            if len(entity_text) > 0:

                # If the entity text is not present in text space
                if entity_text not in text_space:

                    # Set the number of errors to zero.
                    e = 0

                    # For now, the text to match is none.
                    matched_text = None

                    # As the SROIE dataset contains so many errors,
                    # the idea here is to permit at most e, e + 1, e + 2, ..., n errors
                    # until the extraction of the entity value is correct.
                    while matched_text is None and e <= n_errors:
                        matched_text = regex.search("(?e)(" + entity_text + "){e<=" + str(e) + "}", text_space)
                        e += 1

                    # Extract the new entity text, i.e., the one which contains errors.
                    entity_text = matched_text.group().strip()

                # Split this new entity text by the new line character (if any).
                word_groups = entity_text.split("\n")

                # Assign the entity text to its corresponding class.
                for word_group in word_groups:
                    for k, text_line in enumerate(text_lines):
                        is_none_class = entity_classes[k] == none_class
                        if is_none_class:
                            text_line = text_line.strip()
                            word_group = word_group.strip()
                            if len(word_group) != 0 and word_group in text_line:
                                entity_classes[k] = klass
                                if entity != "total":
                                    break

    return entity_classes


def assign_labels(ocr_data: list, entities: dict, classes_labels: dict):
    """
    Given a txt path_to_file containing the OCR data, this will assign a label to each line.
    
    Args:
        ocr_data (list): A list of dictionaries for which each line is assigned to a given label.
        entities (dict): The corresponding json path_to_file to the txt path_to_file containing the OCR data.
        classes_labels (dict): The classes (key) and labels (value) dictionary.

    Returns:
        A list of dictionaries where each line is assigned to a label.
        
    """

    # Create the text space, i.e., concatenate all the OCR data lines into one string.
    text_space = ""
    for data_line in ocr_data:
        text_space += data_line["text"] + "\n"
    text_space = text_space.strip()

    entity_classes = compute_entity_classes(text_space=text_space, entities=entities)

    max_length = len(ocr_data)

    # For each data line, which is a dict containing the filename,
    # the four bounding box coordinates and the text in a receipt
    for i in range(max_length):
        ith_data_line = ocr_data[i]

        klass = entity_classes[i]

        assigned_label = classes_labels[klass]

        ith_data_line["label"] = assigned_label

        ith_data_line["class"] = klass

        # If the assigned label is 'total'
        if assigned_label == "total":
            # then the idea here is to label the total if and only if it is preceded by a certain group of words.
            # This pattern includes words like 'TOTAL', 'AMOUNT', etc. Otherwise we label this line as 'none'.
            total_pattern = r"^(.*(TOTAL|AMOUNT|DUE|AMT|ROUND|RND|RM|GST)).*"
            text = ith_data_line["text"].strip()
            ith_line_to_match = regex.search(total_pattern, text)
            if ith_line_to_match is None:
                n = 1
                j = i - 1
                found = False
                while j >= 0 and n < 3:
                    jth_data_line = ocr_data[j]
                    text = jth_data_line["text"].strip()
                    jth_line_to_match = regex.search(total_pattern, text)
                    jth_line_not_to_match = regex.search(r"^(?!.*(CASH|QTY|TAX|INVOICE)).*", text)
                    if jth_line_not_to_match is None:
                        break
                    elif jth_line_to_match is not None:
                        found = True
                    j -= 1
                    n += 1

                if not found:
                    ith_data_line["label"] = "none"
                    ith_data_line["class"] = entities["none"][0]

    return ocr_data


def generate_csv_for_training(path_to_csv_dir: str, labels_classes: dict):
    """
    Generate the csv files for the training phase.
    
    Args:
        path_to_csv_dir (string): The path where the csv files will be.
        labels_classes (dict): The labels (keys) and classes (values) dictionary.

    """

    print("\nGenerating the csv path_to_file for training...\n")

    path_to_task1_training_set = osp.join(SROIE_ROOT, "0325updated.task1train(626p)")

    path_to_task2_training_set = osp.join(SROIE_ROOT, "0325updated.task2train(626p)")

    if not osp.exists(path_to_task1_training_set) or not osp.exists(path_to_task1_training_set):
        raise ValueError("The files for the task 1 or 2 do not exist!!")

    task1_files = list(os.scandir(path=path_to_task1_training_set))
    task1_training_files = [get_basename(file.path) for file in task1_files if file.name.endswith("txt")]

    task2_files = list(os.scandir(path=path_to_task2_training_set))
    task2_training_files = [get_basename(file.path) for file in task2_files if file.name.endswith("txt")]

    # Making sure their names coincide.
    train_filenames = sorted(list(set(task2_training_files).intersection(task1_training_files)))

    csv_columns = ["filename", "text", "label", "class"]

    # all_texts = []
    # text_max_length = 0

    for i, train_filename in enumerate(train_filenames):
        txt_file = train_filename + ".txt"
        json_path = osp.join(path_to_task2_training_set, txt_file)
        annotation_path = osp.join(path_to_task1_training_set, txt_file)
        entities = read_json_file(json_path, labels_classes=labels_classes)
        ocr_data = parse_annotation(annotation_path)
        classes_labels = {v: k for k, v in labels_classes.items()}
        new_ocr_data = assign_labels(ocr_data=ocr_data, entities=entities, classes_labels=classes_labels)
        csv_file = osp.join(path_to_csv_dir, train_filename + ".csv")
        try:
            with open(csv_file, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=csv_columns, delimiter=",")
                writer.writeheader()
                for data_line in new_ocr_data:
                    writer.writerow(data_line)
                    # text_line = data_line["text"].strip()
                    # current_text_max_length = len(text_line)
                    # text_max_length = max(text_max_length, current_text_max_length)
                    # all_texts.append(text_line)
        except IOError:
            print("I/O error")

    # print("Generating the vocabulary for training...\n")
    # parent_dir = osp.abspath(osp.join(path_to_csv_dir, os.pardir))
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
    #     with open(path_to_file=txt_file, mode='w') as f:
    #         f.write(str(text_max_length))
    # except IOError:
    #     print("I/O error")


def generate_csv_for_evaluation(test_folder: str):
    """
    Generate the csv files for the test/evaluation phase.
    
    Args:
        test_folder (string): The path where the csv files will be.
        
    """

    print("\nGenerating the csv path_to_file for the evaluation...\n")

    path_to_task2_annotations = osp.join(SROIE_ROOT, "text.task1_2-test（361p)")

    path_to_task3_test_set = osp.join(SROIE_ROOT, "task3-test 347p) -", "task3-test（347p)")

    if not osp.isdir(path_to_task2_annotations) or not osp.isdir(path_to_task3_test_set):
        raise ValueError("The files for task 3 do not exist!")

    task3_jpg_files = list(os.scandir(path=path_to_task3_test_set))
    task3_filenames = [get_basename(file.path) for file in task3_jpg_files if file.name.endswith("jpg")]

    task2_txt_files = list(os.scandir(path=path_to_task2_annotations))
    task2_filenames = [get_basename(file.path) for file in task2_txt_files if file.name.endswith("txt")]

    test_filenames = list(set(task3_filenames).intersection(task2_filenames))

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

    import argparse

    from keyword_information_extraction.configs import configs
    from keyword_information_extraction.data.dataset.constant_variables import TOTAL_PATTERN, DATE_PATTERN_1, \
        DATE_PATTERN_2

    parser = argparse.ArgumentParser(description="Pre-processing step",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config-file", action="store", help="The path to the configs path_to_file.")

    args = parser.parse_args()

    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs path_to_file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()

    labels_classes = dict(configs.DATASET.LABELS_CLASSES)
    labels_classes = dict(sorted(labels_classes.items(), key=lambda item: item[1]))
    classes_labels = {v: k for k, v in labels_classes.items()}
    print("Labels & classes: ", labels_classes)
    print()
    print("Classes & labels: ", classes_labels)
    print()

    json_path = osp.join(SROIE_ROOT, "0325updated.task2train(626p)", "X00016469612.txt")
    json_file = read_json_file(json_path, labels_classes=labels_classes)
    print("Entities: ", json_file)
    print()

    annotation_path = osp.join(SROIE_ROOT, "0325updated.task1train(626p)", "X00016469612.txt")
    ocr_data = parse_annotation(annotation_path)
    print("OCR data:", ocr_data)
    print()

    new_ocr_data = assign_labels(ocr_data=ocr_data, entities=json_file, classes_labels=classes_labels)
    line_number = 1
    text_line = ocr_data[line_number]["text"]
    label = ocr_data[line_number]["label"]
    print("line2Match:", text_line)
    print("label: ", label)
    print()

    csv_columns = ["filename", "text", "label", "class"]
    path_to_csv = osp.join("keyword_information_extraction/data/preprocessing/csv")
    if not osp.exists(path_to_csv):
        os.makedirs(path_to_csv)
    csv_file = osp.join(path_to_csv, "X00016469612.csv")
    try:
        with open(csv_file, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns, delimiter=",")
            writer.writeheader()
            for data_line in new_ocr_data:
                writer.writerow(data_line)
    except IOError:
        print("I/O error")

    str2Matches = ["- 75.00", "RM85.00", "$8.55", "RM 65.00", "85.00SR", "46.89", "AMOUNT DUE 30.25", "1.0"]
    for str2Match in str2Matches:
        matched = regex.search(TOTAL_PATTERN, str2Match.strip())
        if matched is not None:
            print("Could matched {0}".format(matched.group().strip()))
        else:
            print("Could not matched {0}".format(str2Match))
    print()

    str2Matches = ["DATE: 2018-03-23", ": 2018-04-06", "06/04/18", "3004 STORED 22 MAR 18 02:1"]
    for str2Match in str2Matches:
        m = regex.search(DATE_PATTERN_1, str2Match.strip())
        if m is None:
            m = regex.search(DATE_PATTERN_2, str2Match.strip())
        if m is not None:
            print("Could matched {0}".format(m.group()))
        else:
            print("Could not matched {0}".format(str2Match))

    m = regex.search("(?e)(dok){e<=1}", "cat and dog")
    if m is not None:
        print(m.group())
        print(m.fuzzy_counts)
