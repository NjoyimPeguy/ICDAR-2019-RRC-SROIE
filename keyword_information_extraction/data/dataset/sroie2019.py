import os
import csv
import math
import torch
import numpy as np
import os.path as osp

from typing import Union
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .constant_variables import VOCAB, MAXIMUM_LENGTH

from functional.utils.dataset import compute_class_weights

from keyword_information_extraction.utils.misc import get_basename
from keyword_information_extraction.data.preprocessing import generate_csv_for_training, generate_csv_for_evaluation


class SROIE2019Dataset(Dataset):
    def __init__(self, directory_dict: dict, labels_classes: Union[dict, None] = None):
        """
        An abstract representation of the SROIE 2019 dataset.

        Args:
            directory_dict (dict): A dictionary of the new directory, where the csv path_to_file will be generated.
            labels_classes (dict or None): The labels (keys) & classes (values) dictionary.

        """

        data_dir = directory_dict["data-dir"]

        split = directory_dict["split"]

        splits = ("test", "train")

        if split not in splits:
            raise ValueError("The split provided is not supported! It must be one the {0}".format(splits))

        self.split: str = split

        path_to_csv_dir = osp.join(data_dir, split)

        if not osp.exists(path_to_csv_dir):
            os.makedirs(path_to_csv_dir)

        if len(os.listdir(path_to_csv_dir)) == 0:  # if the directory is empty...
            if split == "test":
                generate_csv_for_evaluation(path_to_csv_dir)
            else:
                generate_csv_for_training(path_to_csv_dir=path_to_csv_dir, labels_classes=labels_classes)

        # voc_file = osp.join(osp.abspath(osp.join(path_to_csv_dir, os.pardir)), "train-vocabulary.txt")
        # with open(path_to_file=voc_file, mode='r') as f:
        #     self.vocabulary = str(f.readline())

        # max_length_file = osp.join(osp.abspath(osp.join(path_to_csv_dir, os.pardir)), "text-max-length.txt")
        # with open(path_to_file=max_length_file, mode='r') as f:
        #     max_length = f.readline()
        #     self.text_max_length = int(max_length)

        self.vocabulary: str = VOCAB

        self.text_max_length: str = MAXIMUM_LENGTH

        csv_file_paths = os.scandir(path=path_to_csv_dir)
        self.csv_filenames: list = [get_basename(csv_file.path) for csv_file in csv_file_paths]

        self.classes = []
        self.raw_texts = []
        self.one_hot_text_encodings = []

        all_classes = []

        # For each csv path_to_file.
        for csv_filename in self.csv_filenames:

            curr_texts = []

            one_hot_texts = []

            one_hot_classes = []

            path_to_csv_file = osp.join(path_to_csv_dir, csv_filename + ".csv")

            with open(file=path_to_csv_file, mode="r") as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=",")

                # Skips the first row of the CSV path_to_file.
                headers = next(csv_reader)

                header_dict = {k: v for v, k in enumerate(headers)}

                for line in csv_reader:

                    if len(line) != 0:

                        raw_text = line[header_dict["text"]].strip()

                        if split == "test":
                            curr_texts.append(raw_text)
                        else:
                            class_idx = int(line[header_dict["class"]])
                            one_hot_classes.append(class_idx)

                        one_hot_texts.append(self.text2Array(raw_text=raw_text.upper()))

            if split == "test":
                self.raw_texts.append(curr_texts)
            else:
                all_classes.extend(one_hot_classes)
                self.classes.append(torch.tensor(one_hot_classes, dtype=torch.int64))

            self.one_hot_text_encodings.append(torch.from_numpy(np.array(one_hot_texts, dtype=np.float32)))

        if split == "train":
            all_classes = np.array(all_classes, dtype=np.int64)
            # [1.0, 2.0, 2.1, 1.1, 1.5]
            self.class_weights = np.round(compute_class_weights(classes=all_classes), decimals=1).astype(np.float32)

    def __len__(self):
        return len(self.csv_filenames)

    def __getitem__(self, index):

        one_hot_texts = self.one_hot_text_encodings[index]

        if self.split != "test":
            output = (one_hot_texts, self.classes[index])
        else:
            output = (one_hot_texts, self.raw_texts[index], self.csv_filenames[index])

        return output

    def text2Array(self, raw_text):
        """
        Convert a raw text to a one-hot character vectors.

        Args:
            raw_text: The input text.

        Returns:
            An array of one-hot character vectors.

        """
        if len(raw_text) == 0:
            raise ValueError("Cannot have an empty text!")

        data = []

        for i, char in enumerate(raw_text):
            letter2idx = self.vocabulary.find(char) + 1  # +1 to avoid the confusion with the token padding value.
            data.append(letter2idx)

        data = np.array(data, dtype=np.float32)

        # the length of the text array must be at most text max length
        if len(data) > self.text_max_length:
            data = data[:self.text_max_length]
        elif 0 < len(data) < self.text_max_length:
            data = np.concatenate((data, np.zeros((self.text_max_length - len(data),), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.text_max_length,), dtype=np.float32)

        return data


class TrainBatchCollator:
    def __init__(self, class_labels_padding_value: int):
        """
        The collate function to use in data loader when training.
        It merges a list of samples to form a mini-batch of Tensor(s).

        Args:
            class_labels_padding_value (int): The class label padding value to use in pad sequence function.
            This value will be ignored when computing the loss with the cross entropy loss.
        """
        self.class_labels_padding_value = class_labels_padding_value

    def __call__(self, batch_samples):
        class_labels = []

        one_hot_texts = []

        for batch_sample in batch_samples:
            text_data, classes = batch_sample

            class_labels.append(classes)

            one_hot_texts.append(text_data)

        one_hot_texts = pad_sequence(one_hot_texts, batch_first=True, padding_value=0)

        class_labels = pad_sequence(class_labels, batch_first=True, padding_value=self.class_labels_padding_value)

        return one_hot_texts, class_labels


class TestBatchCollator:
    def __call__(self, batch_samples):
        raw_texts = []

        csv_filenames = []

        one_hot_texts = []

        for batch_sample in batch_samples:
            text_data, raw_text, csv_filename = batch_sample

            raw_texts.append(raw_text)

            one_hot_texts.append(text_data)

            csv_filenames.append(csv_filename)

        one_hot_texts = pad_sequence(one_hot_texts, batch_first=True, padding_value=0)

        return one_hot_texts, raw_texts, csv_filenames