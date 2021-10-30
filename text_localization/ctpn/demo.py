import os
import sys
import cv2
import time
import copy
import torch
import argparse
import numpy as np
import os.path as osp

sys.path.append(os.getcwd())

from PIL import Image
from vizer.draw import draw_boxes
from text_localization.ctpn.model import CTPN
from text_localization.ctpn.configs import configs
from text_localization.ctpn.utils.dset import read_image
from text_localization.ctpn.utils.logger import create_logger
from text_localization.ctpn.data.postprocessing import TextDetector, remove_empty_boxes
from text_localization.ctpn.data.preprocessing.augmentation import BasicDataTransformation
from text_localization.ctpn.data.preprocessing.transformations import ToSobelGradient, ToMorphology, \
    CropImage, ConvertColor

parser = argparse.ArgumentParser(description="CTPN: prediction phase")

parser.add_argument("--config-file", action="store", help="The path to the configs file.")
parser.add_argument("--trained-model", action="store", help="The path to the trained model state dict file")
parser.add_argument("--image-folder", default="./text_localization/demo/images", type=str,
                    help="The path to the trained model state dict file")
parser.add_argument("--output-folder", default="./text_localization/demo/results/ctpn", type=str,
                    help="The directory to save prediction results")
parser.add_argument("--remove-extra-white", action="store_true",
                    help="Enable or disable the need for removing the extra white space on the scanned receipts."
                         "By default it is disable.")
parser.add_argument("--use-cuda", action="store_true",
                    help="enable/disable cuda during prediction. By default it is disable")
parser.add_argument("--use-amp", action="store_true",
                    help="Enable or disable the automatic mixed precision. By default it is disable."
                         "For further info, check those following links:"
                         "https://pytorch.org/docs/stable/amp.html"
                         "https://pytorch.org/docs/stable/notes/amp_examples.html"
                         "https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html")
parser.add_argument("--gpu-device", default=0, type=int, help="Specify the GPU ID to use. By default, it is the ID 0")

args = parser.parse_args()


class Prediction:
    def __init__(self):

        possible_extension_image = ("jpg", "png", "jpeg", "JPG")

        files = list(sorted(os.scandir(path=args.image_folder), key=lambda f: f.name))
        self.images = [f for f in files if f.name.endswith(possible_extension_image)]

        if len(self.images) == 0:
            raise ValueError("There are no images for prediction!")

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

        if not osp.exists(args.output_folder):
            os.makedirs(args.output_folder)

        output_dir = os.path.normpath(configs.OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.logger = create_logger(name="Prediction phase", output_dir=output_dir, log_filename="prediction-log")

        # Initialisation and loading the model's weight
        modelArgs = dict(configs.MODEL.ARGS)
        self.model = CTPN(**modelArgs)
        checkpoint = torch.load(args.trained_model, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Putting the trained model into the right device
        self.model = self.model.to(self.device)

        self.text_detector = TextDetector(configs=configs)

        self.basic_transform = BasicDataTransformation(configs)

        if args.remove_extra_white:
            # A set of classes for removing extra white space.
            self.cropImage = CropImage()
            self.morphologyEx = ToMorphology()
            self.grayColor = ConvertColor(current="RGB", transform="GRAY")
            self.sobelGradient = ToSobelGradient(cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    @torch.no_grad()
    def run(self):

        self.model = self.model.eval()

        for image in self.images:

            start = time.time()

            original_image_path = image.path

            original_image = np.array(read_image(original_image_path))

            image = copy.deepcopy(original_image)

            # Apply the the same crop logic as in the preprocessing step.
            cropped_pixels_width = cropped_pixels_height = 0
            if args.remove_extra_white and image.shape[1] > 990:
                gray_image = self.grayColor(image)[0]
                threshed_image = self.sobelGradient(gray_image)
                morpho_image = self.morphologyEx(threshed_image)
                image, cropped_pixels_width, cropped_pixels_height = self.cropImage(morpho_image, image)

            original_image_height, original_image_width = image.shape[:2]

            image = self.basic_transform(image, None)[0]

            new_image_height, new_image_width = image.shape[1:]

            image = image.to(self.device)

            image = image.unsqueeze(0)  # Shape: [1, channels, height, width]

            load_time = time.time() - start

            start = time.time()

            # Forward pass using AMP if it is set to True.
            # autocast may be used by itself to wrap inference or evaluation forward passes.
            # GradScaler is not necessary.
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.model(image)

            inference_time = time.time() - start

            # Detections
            detections = self.text_detector(predictions, image_size=(new_image_height, new_image_width))
            detected_bboxes, detected_scores = detections

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

            self.logger.info("Loading time: {lt:.3f} ms || "
                             "Inference: {it:.3f} ms || "
                             "FPS: {fps:.3f} || "
                             "Objects detected: {objects}\n".format(
                lt=round(load_time * 1000),
                it=round(inference_time * 1000),
                fps=round(1.0 / inference_time),
                objects=len(detected_bboxes)
            ))

            # Drawing the bounding boxes on the original image.
            drawn_image = draw_boxes(image=original_image, boxes=detected_bboxes)

            # Saving the drawn image.
            image_name, image_ext = os.path.splitext(os.path.basename(original_image_path))
            Image.fromarray(drawn_image).save(os.path.join(args.output_folder, image_name + image_ext))

            # Writing the annotation .txt file
            with open(os.path.join(args.output_folder, image_name + ".txt"), "w") as f:
                for j, coords in enumerate(detected_bboxes):
                    line = ",".join(str(round(coord)) for coord in coords)
                    line += ", {0}".format(detected_scores[j])
                    line += "\n"
                    f.write(line)


if __name__ == '__main__':

    # Guarding against bad arguments.

    if args.trained_model is None:
        raise ValueError("The path to the trained model is not provided!")
    elif not osp.isfile(args.trained_model):
        raise ValueError("The path to the trained model is wrong!")

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

    Prediction().run()
