import os
import csv
import torch
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .variables import SROIE_VOCAB, SROIE_TEXT_MAX_LENGTH
from sklearn.utils.class_weight import compute_class_weight
from keyword_information_extraction.utils import get_basename
from keyword_information_extraction.data.preprocessing import generate_csv_for_training, generate_csv_for_evaluation


class SROIE2019Dataset(Dataset):
    def __init__(self, new_dir: dict, class_dict=None, validation_args=None):
        """
        An abstract representation of the SROIE 2019 dataset.
        
        Args:
            new_dir: A dictionary of the new directory, where the csv file will generated.
            class_dict: The class dictionary.
            validation_args: The validation arguments.
            
        """

        data_dir = new_dir["data-dir"]
        split = new_dir["split"]
        splits = ("test", "train", "val")
        if split not in splits:
            raise ValueError("The split provided is not supported! It must be one the {0}".format(splits))

        path_to_csv_dir = osp.join(data_dir, split)

        if split == "train":
            if osp.exists(path_to_csv_dir):
                os.system("rm -rf " + path_to_csv_dir)
            os.makedirs(path_to_csv_dir)

        if split == "test" and not osp.exists(path_to_csv_dir):
            os.makedirs(path_to_csv_dir)

        if len(os.listdir(path_to_csv_dir)) == 0:  # if the directory is empty...
            if split == "test":
                generate_csv_for_evaluation(path_to_csv_dir)
            elif split != "val":
                generate_csv_for_training(save_folder=path_to_csv_dir,
                                          class_dict=class_dict,
                                          validation_args=validation_args)

        # voc_file = osp.join(osp.abspath(osp.join(path_to_csv_dir, os.pardir)), "train-vocabulary.txt")
        # with open(file=voc_file, mode='r') as f:
        #     self.vocabulary = str(f.readline())

        # max_length_file = osp.join(osp.abspath(osp.join(path_to_csv_dir, os.pardir)), "text-max-length.txt")
        # with open(file=max_length_file, mode='r') as f:
        #     max_length = f.readline()
        #     self.text_max_length = int(max_length)

        self.vocabulary = SROIE_VOCAB

        self.text_max_length = SROIE_TEXT_MAX_LENGTH

        self.split = split
        csv_file_paths = list(sorted(os.scandir(path=path_to_csv_dir), key=lambda file: file.name))
        self.csv_filenames = [get_basename(csv_file.path) for csv_file in csv_file_paths]
        self.raw_texts = []
        self.class_labels = []
        self.class_weights = []
        self.one_hot_text_encodings = []

        for csv_filename in self.csv_filenames:
            curr_texts = []
            one_hot_texts = []
            one_hot_labels = []
            path_to_csv_file = osp.join(path_to_csv_dir, csv_filename + ".csv")
            with open(file=path_to_csv_file, mode="r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                # This will skip the first row of the CSV file.
                headers = next(csv_reader)
                header_dict = {k: v for v, k in enumerate(headers)}
                for line in csv_reader:
                    if len(line) != 0:
                        raw_text = line[header_dict["text"]].strip()
                        if split == "test":
                            curr_texts.append(raw_text)
                        else:
                            class_idx = int(line[header_dict["class"]])
                            one_hot_labels.append(class_idx)
                        one_hot_texts.append(self.text2Array(raw_text=raw_text.upper()))
            if split == "test":
                self.raw_texts.append(curr_texts)
            else:
                classes = torch.tensor(one_hot_labels, dtype=torch.int64)
                self.class_labels.append(classes)
                unique_labels = np.unique(one_hot_labels)
                class_weight = compute_class_weight(class_weight="balanced",
                                                    classes=unique_labels,
                                                    y=np.array(one_hot_labels))
                weights = {k: v for k, v in zip(unique_labels, class_weight)}
                class_weight = torch.zeros(size=classes.shape, dtype=torch.float32)
                for klass, weight in weights.items():
                    class_weight[classes == klass] = weight
                self.class_weights.append(class_weight)
            self.one_hot_text_encodings.append(torch.from_numpy(np.array(one_hot_texts, dtype=np.int64)))

    def __len__(self):
        return len(self.csv_filenames)

    def __getitem__(self, index):
        one_hot_texts = self.one_hot_text_encodings[index]
        if self.split == "test":
            output = (one_hot_texts, self.raw_texts[index], self.csv_filenames[index])
        else:
            output = (one_hot_texts, self.class_labels[index], self.class_weights[index])
        return output

    def text2Array(self, raw_text):
        """
        Convert a raw text to a one-hot character vectors.
        
        Args:
            raw_text: The input text.

        Returns:
            An array of one-hot character vectors whose shape is (text_max_length,)
            
        """
        if len(raw_text) == 0:
            raise ValueError("Cannot have an empty text!")

        data = []
        for i, char in enumerate(raw_text):
            letter2idx = self.vocabulary.find(char) + 1  # +1 to avoid the confusion with the token padding value
            data.append(letter2idx)
        data = np.array(data, dtype=np.int64)

        # the length of the text array must be at most max_length
        if len(data) > self.text_max_length:
            data = data[:self.text_max_length]
        elif 0 < len(data) < self.text_max_length:
            data = np.concatenate((data, np.zeros((self.text_max_length - len(data),), dtype=np.int64)))
        elif len(data) == 0:
            data = np.zeros((self.text_max_length,), dtype=np.int64)
        return data


class TrainBatchCollator:
    def __init__(self, class_labels_padding_value: int):
        """
        The collate function to use in data loader. It merges a list of samples to form a mini-batch of Tensor(s).
        
        Args:
            class_labels_padding_value: The class label padding value to use in pad sequence function.
        """
        self.class_labels_padding_value = class_labels_padding_value

    def __call__(self, batch_samples):
        one_hot_texts = []
        class_labels = []
        class_weights = []
        for batch_sample in batch_samples:
            text_data, classes, class_weight = batch_sample
            one_hot_texts.append(text_data)
            class_labels.append(classes)
            class_weights.append(class_weight)
        one_hot_texts = pad_sequence(one_hot_texts, batch_first=True, padding_value=0)
        class_labels = pad_sequence(class_labels, batch_first=True, padding_value=self.class_labels_padding_value)
        class_weights = pad_sequence(class_weights, batch_first=True, padding_value=0)
        return one_hot_texts, class_labels, class_weights


class TestBatchCollator:
    def __call__(self, batch_samples):
        one_hot_texts = []
        raw_texts = []
        csv_filenames = []
        for batch_sample in batch_samples:
            text_data, raw_text, csv_filename = batch_sample
            one_hot_texts.append(text_data)
            raw_texts.append(raw_text)
            csv_filenames.append(csv_filename)
        one_hot_texts = pad_sequence(one_hot_texts, batch_first=True, padding_value=0)
        return one_hot_texts, raw_texts, csv_filenames
