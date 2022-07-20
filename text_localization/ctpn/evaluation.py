import os
import sys

sys.path.append(os.getcwd())

import cv2
import copy
import torch
import warnings
import numpy as np
import os.path as osp

from options import TestArgs
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.datasets.dataset_roots import SROIE_ROOT

from functional.event_tracker import Logger
from functional.utils.dataset import read_image
from functional.utils.box import remove_empty_boxes
from functional.data.transformation.computer_vision import ToSobelGradient, ToMorphology, CropImage, ConvertColor

from text_localization.ctpn.model import CTPN
from text_localization.ctpn.configs import configs
from text_localization.ctpn.data.postprocessing import TextDetector
from text_localization.ctpn.data.augmentation import BasicDataTransformation

# Put warnings to silence.
warnings.filterwarnings("ignore")

test_args = TestArgs(description="Text localization: evaluation")
parser = test_args.get_parser()
parser.add_argument("--remove-extra-white", action="store_true",
                    help="Enable or disable the need to crop the extra white background. "
                         "By default, there is not any removal of the white background.")
args, _ = parser.parse_known_args()


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
        self.use_amp: bool = args.use_amp and use_cuda

        # The declaration and tensor type of the CPU/GPU device.
        if not use_cuda:
            self.device: torch.device = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            self.device: torch.device = torch.device("cuda")
            torch.cuda.set_device(args.gpu_device)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

            # Enabling cuDNN.
            torch.backends.cudnn.enabled = True

        output_dir = os.path.normpath(configs.OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.logger: Logger = Logger(name="Evaluation phase", output_dir=output_dir, log_filename="evaluation-log")

        self.logger.info("Loading the model's weights.")
        model_args = dict(configs.MODEL.ARGS)
        self.trained_model: CTPN = CTPN(**model_args)
        checkpoint = torch.load(args.model_checkpoint, map_location=torch.device("cpu"))
        model_state_dict = checkpoint.get("model_state_dict")
        if model_state_dict is None:
            raise ValueError("Impossible to get the key '{0}' from the checkpoint!".format("model_state_dict"))
        self.trained_model.load_state_dict(model_state_dict)

        # Putting the trained model into the right device
        self.trained_model = self.trained_model.to(self.device, non_blocking=True)

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

        nb_images = len(self.images)

        self.logger.info("About to evaluate {0} images.\n".format(nb_images))

        for i, image in enumerate(self.images):

            image_id = osp.splitext(image.name)[0]

            original_image = np.array(read_image(image.path))

            image = copy.deepcopy(original_image)

            # Apply the same crop logic as in the preprocessing step.
            cropped_pixels_width = cropped_pixels_height = 0

            if args.remove_extra_white and image.shape[1] > 990:
                gray_image = self.grayColor(image)[0]
                threshed_image = self.sobelGradient(gray_image)
                morpho_image = self.morphologyEx(threshed_image, erode_iterations=6, dilate_iterations=6)
                image, cropped_pixels_width, cropped_pixels_height = self.cropImage(morpho_image, image)

            original_image_height, original_image_width = image.shape[:2]

            self.logger.info("{0}/{1}: evaluating image name: {2}.\n".format(i + 1, nb_images, image_id))

            image = self.basic_augmentation(image)[0]

            new_image_height, new_image_width = image.shape[1:]

            image = image.to(self.device, non_blocking=True)

            image = image.unsqueeze(0)  # Shape: (1, C, H, W)

            # Forward pass using AMP if it is set to True.
            # autocast may be used by itself to wrap inference or task1 forward passes.
            # GradScaler is not necessary.
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.trained_model(image)

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
                for j, coordinates in enumerate(detected_bboxes):
                    xmin, ymin, xmax, ymax = coordinates
                    # These commented lines below are floats to adjust the bounding box coordinates as the evaluation
                    # protocol is very sensitive to the area of a given box.
                    w = xmax - xmin
                    h = ymax - ymin
                    dw = w * 0.01
                    dh = h * 0.05
                    box = [xmin - dw, ymin - dh, xmax + dw, ymin - dh, xmax + dw, ymax + dh, xmin - dw, ymax + dh]
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
        os.system("chmod a+x scripts/evaluation/task1/script.py")
        os.system("scripts/evaluation/task1/script.py " + "-g=" + self.path_to_gt_zip + " -s=" + path_to_submit_zip)


if __name__ == "__main__":

    # Guarding against bad arguments.
    if args.model_checkpoint is None:
        raise ValueError("The path to the trained model is not provided!")
    elif not osp.isfile(args.model_checkpoint):
        raise ValueError("The trained model path_to_file provided is wrong!")

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

    Evaluation().run()
