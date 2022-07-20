import os
import sys

sys.path.append(os.getcwd())

import json
import torch
import warnings
import os.path as osp

# importing useful methods/functions
from functional.data.dataloader import Dataloader
from functional.event_tracker import logger

from options import TestArgs
from zipfile import ZipFile, ZIP_DEFLATED
from keyword_information_extraction.configs import configs
from keyword_information_extraction.data.postprocessing import convert_predictions_to_dict
from keyword_information_extraction.data.dataset import SROIE2019Dataset, TestBatchCollator
from keyword_information_extraction.model.charlm import CharacterLevelCNNHighwayBiLSTM as CharLM

# Put warnings to silence.
warnings.filterwarnings("ignore")

test_args = TestArgs(description="Keyword Information Extraction: evaluation")
parser = test_args.get_parser()
args, _ = parser.parse_known_args()


class Prediction:
    def __init__(self):

        # A boolean to check whether the user is able to use cuda or not.
        use_cuda = torch.cuda.is_available() and args.use_cuda

        # The declaration and tensor type of the CPU/GPU device.
        if not use_cuda:
            self.device: torch.device = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            self.device: torch.device = torch.device("cuda")
            torch.cuda.set_device(args.gpu_device)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

            torch.backends.cudnn.enabled = True

        output_dir = os.path.normpath(configs.OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.path_to_submit: str = osp.join(output_dir, "submit")
        os.system("rm -rf " + self.path_to_submit)
        os.makedirs(self.path_to_submit)

        self.logger = logger(name="Test phase", output_dir=output_dir, log_filename="test-log")

        self.logger.info("Creating the test dataset...")

        test_dataset = SROIE2019Dataset(directory_dict=dict(configs.DATASET.TEST))

        self.nb_images: int = len(test_dataset)

        # The declaration of the dataloader arguments.
        dataloader_args = dict(configs.DATALOADER.EVALUATION)

        dataloader_args["generator"] = torch.Generator(device=self.device)

        # Adding the collate_fn.
        dataloader_args["collate_fn"] = TestBatchCollator()

        self.test_loader = Dataloader(dataset=test_dataset, is_train=False, **dataloader_args)

        self.logger.info("Initialisation and loading the model's weight...")

        model_args = dict(configs.MODEL.ARGS)
        self.trained_model = CharLM(n_classes=configs.DATASET.NUM_CLASSES,
                                    max_seq_length=test_dataset.text_max_length,
                                    char_vocab_size=len(test_dataset.vocabulary),
                                    **model_args)
        checkpoint = torch.load(args.model_checkpoint, map_location=torch.device("cpu"))
        self.trained_model.load_state_dict(checkpoint["model_state_dict"])

        # Putting the trained model into the right device
        self.trained_model = self.trained_model.to(self.device, non_blocking=True)

    @torch.no_grad()
    def run(self):

        self.trained_model = self.trained_model.eval()

        list_raw_texts = []
        list_csv_filenames = []

        probabilities = []
        predicted_classes = []

        for i, batch_samples in enumerate(self.test_loader):
            one_hot_texts, raw_texts, csv_filenames = batch_samples

            list_raw_texts.extend(raw_texts)

            list_csv_filenames.extend(csv_filenames)

            # text data shape: [N, L, C],
            # where N: the number of rows containing in one csv path_to_file, L:input length, C: vocab size
            one_hot_texts = one_hot_texts.to(self.device)

            outputs = self.trained_model(one_hot_texts)

            batch_probs, batch_preds = torch.max(torch.softmax(outputs, dim=2), dim=2)

            probabilities.extend(batch_probs.squeeze().tolist())

            predicted_classes.extend(batch_preds.squeeze().tolist())

        labels_classes = dict(configs.DATASET.LABELS_CLASSES)

        for i, (probs, preds) in enumerate(zip(probabilities, predicted_classes)):
            raw_texts = list_raw_texts[i]
            csv_filename = list_csv_filenames[i]

            results = convert_predictions_to_dict(labels_classes=labels_classes,
                                                  raw_texts=raw_texts,
                                                  probabilities=probs,
                                                  predicted_classes=preds)

            self.logger.info("{0}/{1}: converting image {2} to json.".format(i + 1, self.nb_images, csv_filename))

            with open(osp.join(self.path_to_submit, csv_filename + ".txt"), "w", encoding="utf-8") as json_opened:
                json.dump(results, json_opened, indent=4)

        self.logger.info("Creating the submit zip path_to_file...")
        path_to_submit_zip = self.path_to_submit + ".zip"
        with ZipFile(path_to_submit_zip, "w", ZIP_DEFLATED) as zf:
            for txt_file in os.scandir(self.path_to_submit):
                zf.write(txt_file.path, osp.basename(txt_file.path))


if __name__ == '__main__':

    # Guarding against bad arguments.

    if args.model_checkpoint is None:
        raise ValueError("The path to the trained model is not provided!")
    elif not osp.isfile(args.model_checkpoint):
        raise ValueError("The path to the trained model is wrong!")

    gpu_devices = list(range(torch.cuda.device_count()))
    if len(gpu_devices) != 0 and args.gpu_device not in gpu_devices:
        raise ValueError("Your GPU ID is out of the range! "
                         "You may want to check it with 'nvidia-smi' or "
                         "'Task Manager' for Windows users.")

    if args.use_cuda and not torch.cuda.is_available():
        raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")

    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()

    Prediction().run()
