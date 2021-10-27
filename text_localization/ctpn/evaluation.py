import os
import sys
import cv2
import copy
import torch
import warnings
import argparse
import numpy as np
import os.path as osp

sys.path.append(os.getcwd())

from zipfile import ZipFile, ZIP_DEFLATED
from text_localization.ctpn.model import CTPN
from text_localization.ctpn.configs import configs
from scripts.datasets.dataset_roots import SROIE_ROOT
from text_localization.ctpn.utils.dset import read_image
from text_localization.ctpn.utils.logger import create_logger
from text_localization.ctpn.data.postprocessing import TextDetector, remove_empty_boxes
from text_localization.ctpn.data.preprocessing.augmentation import BasicDataTransformation
from text_localization.ctpn.data.preprocessing.transformations import ToSobelGradient, ToMorphology, \
    CropImage, ConvertColor

# Put warnings to silence.
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="CTPN: evaluation phase.")

parser.add_argument("--config-file", action="store", help="The path to the configs file.")
parser.add_argument("--trained-model", action="store", help="The path to the trained model state dict file")
parser.add_argument("--remove-extra-white", action="store_true",
                    help="Enable or disable the need for removing the extra white space on the scanned receipts."
                         "By default it is disable.")
parser.add_argument("--use-cuda", action="store_true", help="enable/disable cuda training")
parser.add_argument("--use-amp", action="store_true",
                    help="Enable or disable the automatic mixed precision. By default it is disable."
                         "For further info, check those following links:"
                         "https://pytorch.org/docs/stable/amp.html"
                         "https://pytorch.org/docs/stable/notes/amp_examples.html"
                         "https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html")
parser.add_argument("--gpu-device", default=0, type=int, help="Specify the GPU ID to use. By default, it is the ID 0")

args = parser.parse_args()


class Evaluation:
    
    def __init__(self):
        path_to_images = osp.join(SROIE_ROOT, "task1_2_test(361p)")
        if not osp.exists(path_to_images):
            raise ValueError("You have completely forgotten to download the main dataset!")
        self.images = list(sorted(os.scandir(path=path_to_images), key=lambda f: f.name))
        self.path_to_gt_zip = osp.join(configs.OUTPUT_DIR, "gt.zip")
        self.path_to_submit = osp.join(configs.OUTPUT_DIR, "submit")
        if not osp.isdir(self.path_to_submit):
            os.makedirs(self.path_to_submit)
        
        # A boolean to check whether the user is able to use cuda or not.
        use_cuda = torch.cuda.is_available() and args.use_cuda
        
        # A boolean to check whether the user is able to use amp or not.
        self.use_amp = args.use_amp and use_cuda
        
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
        
        self.logger = create_logger(name="Evaluation phase", output_dir=output_dir, log_filename="evaluation-log")
        
        # Initialisation and loading the model's weight
        modelArgs = dict(configs.MODEL.ARGS)
        self.model = CTPN(**modelArgs)
        checkpoint = torch.load(args.trained_model, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Putting the trained model into the right device
        self.model = self.model.to(self.device)
        
        self.text_detector = TextDetector(configs=configs)
        
        self.basic_augmentation = BasicDataTransformation(configs=configs)
        
        if args.remove_extra_white:
            # A set of classes for removing extra white space.
            self.cropImage = CropImage()
            self.morphologyEx = ToMorphology()
            self.grayColor = ConvertColor(current="RGB", transform="GRAY")
            self.sobelGradient = ToSobelGradient(cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    @torch.no_grad()
    def run(self):
        
        self.model = self.model.eval()
        
        nb_images = len(self.images)
        self.logger.info("About to evaluate {0} images.\n".format(nb_images))
        
        for i, image in enumerate(self.images):
            
            image_id = osp.splitext(image.name)[0]
            original_image = np.array(read_image(image.path))
            
            image = copy.deepcopy(original_image)
            
            # Apply the the same crop logic as in the preprocessing step.
            cropped_pixels_width = cropped_pixels_height = 0
            if args.remove_extra_white and image.shape[1] > 990:
                gray_image = self.grayColor(image)[0]
                threshed_image = self.sobelGradient(gray_image)
                morpho_image = self.morphologyEx(threshed_image)
                image, cropped_pixels_width, cropped_pixels_height = self.cropImage(morpho_image, image)
            
            original_image_height, original_image_width = image.shape[:2]
            
            self.logger.info("{0}/{1}: evaluating image name: {2}.\n".format(i + 1, nb_images, image_id))
            
            image = self.basic_augmentation(image, None)[0]
            
            new_image_height, new_image_width = image.shape[1:]
            
            image = image.to(self.device)
            
            image = image.unsqueeze(0)  # Shape: [1, height, width, nb_channels]
            
            # Forward pass using AMP if it is set to True.
            # autocast may be used by itself to wrap inference or evaluation forward passes.
            # GradScaler is not necessary.
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.model(image)
            
            # Detections
            detected_bboxes, _ = self.text_detector(predictions, image_size=(new_image_height, new_image_width))
            
            # Scaling the bounding boxes back to the original image.
            ratio_w = original_image_width / new_image_width
            ratio_h = original_image_height / new_image_height
            size_ = np.array([[ratio_w, ratio_h, ratio_w, ratio_h]])
            detected_bboxes *= size_
            
            # Adjusting the bounding box coordinates, if the images were previously cropped.
            detected_bboxes[:, 0::2] += cropped_pixels_width
            detected_bboxes[:, 1::2] += cropped_pixels_height
            
            # Removing empty bounding boxes.
            qualified_bbox_indices = remove_empty_boxes(original_image, detected_bboxes)
            detected_bboxes = detected_bboxes[qualified_bbox_indices]
            
            with open(osp.join(self.path_to_submit, image_id + ".txt"), "w") as f:
                for j, coords in enumerate(detected_bboxes):
                    xmin, ymin, xmax, ymax = coords
                    box = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                    line = ",".join(str(round(coord)) for coord in box)
                    line += "\n"
                    f.write(line)
        
        path_to_submit_zip = self.path_to_submit + ".zip"
        with ZipFile(path_to_submit_zip, "w", ZIP_DEFLATED) as zf:
            for txt_file in os.scandir(self.path_to_submit):
                zf.write(txt_file.path, osp.basename(txt_file.path))
        
        path_to_annotations = osp.join(SROIE_ROOT, "text.task1_2-test（361p)")
        with ZipFile(self.path_to_gt_zip, "w", ZIP_DEFLATED) as zf:
            for txt_file in os.scandir(path_to_annotations):
                zf.write(txt_file.path, osp.basename(txt_file.path))
        
        self.logger.info("Waiting for the ICDAR script to execute...")
        
        os.system("rm -rf " + self.path_to_submit)
        os.system("chmod a+x scripts/sroie2019/evaluation/script.py")
        os.system("scripts/sroie2019/evaluation/script.py " + "-g=" + self.path_to_gt_zip + " -s=" + path_to_submit_zip)


if __name__ == "__main__":
    
    # Guarding against bad arguments.
    
    if args.trained_model is None:
        raise ValueError("The path to the trained model is not provided!")
    elif not osp.isfile(args.trained_model):
        raise ValueError("The trained model file provided is wrong!")
    
    gpu_devices = list(range(torch.cuda.device_count()))
    if len(gpu_devices) != 0 and args.gpu_device not in gpu_devices:
        raise ValueError("Your GPU ID is out of the range! You may want to check with 'nvidia-smi'")
    elif args.use_cuda:
        if not torch.cuda.is_available():
            raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")
    elif args.use_amp:
        raise ValueError("The arguments --use-cuda, --use-amp and --gpu-device must be used together!")
    
    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()
    
    Evaluation().run()
