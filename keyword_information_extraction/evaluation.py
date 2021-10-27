import os
import sys
import json
import torch
import argparse
import os.path as osp
import torch.nn.functional as F

sys.path.append(os.getcwd())

# importing useful methods/functions from task 1...
from text_localization.ctpn.utils.logger import create_logger
from text_localization.ctpn.data.datasets.dataloader import create_dataloader

from zipfile import ZipFile, ZIP_DEFLATED
from keyword_information_extraction.configs import configs
from keyword_information_extraction.model.charlm import CharacterLevelCNNHighwayBiLSTM
from keyword_information_extraction.data.postprocessing import convert_predictions_to_dict
from keyword_information_extraction.data.datasets.sroie2019 import SROIE2019Dataset, TestBatchCollator

parser = argparse.ArgumentParser(description="Character-Level: Predictions")

parser.add_argument("--config-file", action="store", help="The path to the yaml configs file.")
parser.add_argument("--trained-model", action="store", help="The path to the trained model state dict file.")
parser.add_argument("--use-cuda", action="store_true",
                    help="Enable or disable cuda during prediction. By default it is disable")
parser.add_argument("--gpu-device", default=0, type=int, help="Specify the GPU id to use for the prediction.")

args = parser.parse_args()


class Prediction:
    def __init__(self):
        
        # A boolean to check whether the user is able to use cuda or not.
        use_cuda = torch.cuda.is_available() and args.use_cuda
        
        # The declaration and tensor type of the CPU/GPU device.
        if not use_cuda:
            self.device = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            self.device = torch.device("cuda")
            torch.cuda.set_device(args.gpu_device)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            
            torch.backends.cudnn.enabled = True
        
        output_dir = os.path.normpath(configs.OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        self.path_to_submit = osp.join(output_dir, "submit")
        os.system("rm -rf " + self.path_to_submit)
        os.makedirs(self.path_to_submit)
        
        self.logger = create_logger(name="Test phase", output_dir=output_dir, log_filename="test-log")
        
        self.logger.info("Creating the test dataset...")
        
        test_dataset = SROIE2019Dataset(new_dir=dict(configs.DATASET.TEST))
        
        self.nb_images = len(test_dataset)
        
        # The declaration of the dataloader arguments.
        dataloader_args = dict(configs.DATALOADER.EVALUATION)
        
        dataloader_args["generator"] = torch.Generator(device=self.device)
        
        # Adding the collate_fn.
        dataloader_args["collate_fn"] = TestBatchCollator()
        
        self.test_loader = create_dataloader(dataset=test_dataset, is_train=False, **dataloader_args)
        
        self.logger.info("Initialisation and loading the model's weight...")
        
        model_params = dict(configs.MODEL.PARAMS)
        
        self.trained_model = CharacterLevelCNNHighwayBiLSTM(n_classes=configs.DATASET.NUM_CLASSES,
                                                            max_seq_length=test_dataset.text_max_length,
                                                            char_vocab_size=len(test_dataset.vocabulary),
                                                            **model_params)
        
        checkpoint = torch.load(args.trained_model, map_location=torch.device("cpu"))
        
        self.trained_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Putting the trained model into the right device
        self.trained_model = self.trained_model.to(self.device)
    
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
            # where N: the number of rows containing in one csv file, L:input length, C: vocab size
            one_hot_texts = one_hot_texts.to(self.device)
            
            outputs = self.trained_model(one_hot_texts)
            
            batch_probs, batch_preds = torch.max(F.softmax(outputs, dim=2), dim=2)
            
            probabilities.extend(batch_probs.squeeze().tolist())
            
            predicted_classes.extend(batch_preds.squeeze().tolist())
        
        class_dict = dict(configs.DATASET.CLASS_NAMES)
        
        for i, (probs, preds) in enumerate(zip(probabilities, predicted_classes)):
            raw_texts = list_raw_texts[i]
            csv_filename = list_csv_filenames[i]
            
            results = convert_predictions_to_dict(class_dict=class_dict,
                                                  raw_texts=raw_texts,
                                                  probabilities=probs,
                                                  predicted_classes=preds)
            
            self.logger.info("{0}/{1}: converting image {2} to json.".format(i + 1, self.nb_images, csv_filename))
            
            with open(osp.join(self.path_to_submit, csv_filename + ".txt"), "w", encoding="utf-8") as json_opened:
                json.dump(results, json_opened, indent=4)
        
        self.logger.info("Creating the submit zip file...")
        path_to_submit_zip = self.path_to_submit + ".zip"
        with ZipFile(path_to_submit_zip, "w", ZIP_DEFLATED) as zf:
            for txt_file in os.scandir(self.path_to_submit):
                zf.write(txt_file.path, osp.basename(txt_file.path))


if __name__ == '__main__':
    
    # Guarding against bad arguments.
    
    if args.trained_model is None:
        raise ValueError("The path to the trained model is not provided!")
    elif not osp.isfile(args.trained_model):
        raise ValueError("The path to the trained model is wrong!")
    
    gpu_devices = list(range(torch.cuda.device_count()))
    if len(gpu_devices) != 0 and args.gpu_device not in gpu_devices:
        raise ValueError("Your GPU ID is out of the range! You may want to check with 'nvidia-smi'")
    elif args.use_cuda and not torch.cuda.is_available():
        raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")
    
    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()
    
    Prediction().run()
