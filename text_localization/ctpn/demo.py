import os
import sys

sys.path.append(os.getcwd())

import gc
import cv2
import time
import copy
import torch
import numpy as np
import os.path as osp

from PIL import Image
from options import TestArgs

from functional.event_tracker import Logger
from functional.utils.dataset import read_image, parse_annotations
from functional.utils.box import remove_empty_boxes, draw_bboxes
from functional.data.transformation.computer_vision import ToSobelGradient, ToMorphology, CropImage, ConvertColor

from text_localization.ctpn.model import CTPN
from text_localization.ctpn.configs import configs
from text_localization.ctpn.data.postprocessing import TextDetector
from text_localization.ctpn.data.augmentation import BasicDataTransformation

test_args = TestArgs(description="Text localization: prediction")
parser = test_args.get_parser()
parser.set_defaults(image_folder="./text_localization/demo/images",
                    help="The path to a directory where there are images to use for predictions.")
parser.set_defaults(output_folder="./text_localization/demo/results/",
                    help="The path to a directory where prediction results will be saved.")
parser.add_argument("--remove-extra-white", action="store_true",
                    help="Enable or disable the need to crop the extra white background. "
                         "By default, there is not any removal of the white background.")
args, _ = parser.parse_known_args()


class Prediction:
    def __init__(self):

        possible_extension_image = ("jpg", "png", "jpeg", "JPG")

        files = list(sorted(os.scandir(path=osp.normpath(args.image_folder)), key=lambda f: f.name))
        self.images: list = [f for f in files if f.name.endswith(possible_extension_image)]

        if len(self.images) == 0:
            raise ValueError("There are no images for prediction!")

        # A boolean to check whether the user is able to use cuda or not.
        self.use_cuda: bool = torch.cuda.is_available() and args.use_cuda

        # A boolean to check whether the user is able to use amp or not.
        self.use_amp: bool = args.use_amp and use_cuda

        # The declaration and tensor type of the CPU/GPU device.
        if not self.use_cuda:
            self.device: torch.device = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            self.device: torch.device = torch.device("cuda")
            torch.cuda.set_device(args.gpu_device)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

            # Enabling cuDNN.
            torch.backends.cudnn.enabled = True

        if not osp.exists(args.output_folder):
            os.makedirs(args.output_folder)

        output_dir = os.path.normpath(configs.OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.logger: Logger = Logger(name="Prediction phase", output_dir=output_dir, log_filename="prediction-log")

        self.logger.info("Loading the model's weights.")
        model_args = dict(configs.MODEL.ARGS)
        self.trained_model = CTPN(**model_args)
        checkpoint = torch.load(args.model_checkpoint, map_location=torch.device("cpu"))
        model_state_dict = checkpoint.get("model_state_dict")
        if model_state_dict is None:
            raise ValueError("Impossible to get the key '{0}' from the checkpoint!".format("model_state_dict"))
        self.trained_model.load_state_dict(model_state_dict)

        # Putting the trained model into the right device
        self.trained_model: torch.nn.Module = self.trained_model.to(self.device, non_blocking=True)

        self.text_detector: TextDetector = TextDetector(configs=configs)

        self.basic_augmentation: BasicDataTransformation = BasicDataTransformation(configs=configs)

        # A set of classes for removing extra white space.
        if args.remove_extra_white:
            self.cropImage: CropImage = CropImage()
            self.morphologyEx: ToMorphology = ToMorphology()
            self.grayColor: ConvertColor = ConvertColor(current="RGB", transform="GRAY")
            self.sobelGradient: ToSobelGradient = ToSobelGradient(cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    @torch.no_grad()
    def run(self):

        self.trained_model = self.trained_model.eval()

        # Force the garbage collector to run.
        gc.collect()

        if self.use_cuda:
            # Before starting the training, all the unoccupied cached memory are released.
            torch.cuda.empty_cache()

            # waits for all tasks in the GPU to complete.
            torch.cuda.current_stream(self.device).synchronize()

        for image in self.images:

            # Starting time.
            start_time = time.time()

            original_image_path = image.path

            original_image = np.array(read_image(original_image_path))

            image = copy.deepcopy(original_image)

            # Apply the same crop logic as in the preprocessing step.
            cropped_pixels_width = cropped_pixels_height = 0
            if args.remove_extra_white and image.shape[1] > 990:
                gray_image = self.grayColor(image)[0]
                threshed_image = self.sobelGradient(gray_image)
                morpho_image = self.morphologyEx(threshed_image, erode_iterations=6, dilate_iterations=6)
                image, cropped_pixels_width, cropped_pixels_height = self.cropImage(morpho_image, image)

            original_image_height, original_image_width = image.shape[:2]

            image = self.basic_augmentation(image)[0]

            new_image_height, new_image_width = image.shape[1:]

            image = image.to(self.device)

            image = image.unsqueeze(0)  # Shape: (1, C, H, W)

            if self.use_cuda:
                # waits for all tasks in the GPU to complete
                torch.cuda.current_stream(self.device).synchronize()

            load_time = time.time() - start_time

            start_time = time.time()

            # Forward pass using AMP if it is set to True.
            # autocast may be used by itself to wrap inference or task1 forward passes.
            # GradScaler is not necessary.
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.trained_model(image)

            inference_time = time.time() - start_time

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
            drawn_image = draw_bboxes(image=original_image, bboxes=detected_bboxes)

            # Saving the drawn image.
            image_name, image_ext = os.path.splitext(os.path.basename(original_image_path))
            Image.fromarray(drawn_image).save(os.path.join(args.output_folder, image_name + image_ext))

            # Writing the annotation .txt path_to_file
            with open(os.path.join(args.output_folder, image_name + ".txt"), "w") as f:
                for j, coordinates in enumerate(detected_bboxes):
                    line = ",".join(str(round(coord)) for coord in coordinates)
                    line += ", {0}".format(detected_scores[j])
                    line += "\n"
                    f.write(line)


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

    if not args.use_cuda and args.use_amp:
        raise ValueError("The arguments --use-cuda, --use-amp must be used together!")

    configs.freeze()

    Prediction().run()
