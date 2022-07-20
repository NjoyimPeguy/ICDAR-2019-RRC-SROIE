import os
import sys

sys.path.append(os.getcwd())

import cv2
import numpy as np
import os.path as osp

from typing import Optional
from torch.utils.data import Dataset
from scripts.datasets.dataset_roots import SROIE_ROOT

from functional.data.preprocessing import crop_preprocessing
from functional.utils.box import draw_single_box, draw_bboxes
from functional.utils.dataset import read_image, read_image_ids, parse_annotations

from text_localization.ctpn.data.transformation import TargetTransform
from text_localization.ctpn.data.augmentation import TrainDataTransformation


class SROIE2019Dataset(Dataset):

    def __init__(self, configs: dict, is_train: Optional[bool] = True):
        """
        An abstract representation of the SROIE 2019 dataset.
        
        Args:
            configs (dict): The configuration path_to_file.
            is_train (bool, optional): A boolean indicating whether to return the original images and boxes
                or the augmented ones. Default: True.
            
        """
        super(SROIE2019Dataset, self).__init__()

        self.is_train: bool = is_train

        self.image_id_basename: str = "trainval.txt"

        self.image_dir_name: str = "JPEGImages"

        self.annotation_dir_name: str = "Annotations"

        self.target_transform: object = TargetTransform(configs)

        self.data_transform: object = TrainDataTransformation(configs)

        self.positive_anchor_label: int = configs.ANCHOR.POSITIVE_LABEL

        self.display_anchor_boxes_during_training: bool = configs.ANCHOR.DISPLAY

        source_folder: str = osp.join(SROIE_ROOT, "0325updated.task1train(626p)")
        if not osp.exists(source_folder):
            raise ValueError("You have completely forgotten to download the main dataset!")

        path_to_base_dir: str = osp.join(SROIE_ROOT, "new-task1-train(626p)")
        if not osp.exists(path_to_base_dir):
            os.makedirs(path_to_base_dir)

        if len(os.listdir(path_to_base_dir)) == 0:
            crop_preprocessing(source_folder=source_folder,
                               parent_folder=path_to_base_dir,
                               image_dir_name=self.image_dir_name,
                               annotation_dir_name=self.annotation_dir_name,
                               image_id_names=self.image_id_basename)

        self.path_to_train_dir: str = path_to_base_dir

        self.image_ids: np.ndarray = read_image_ids(self.path_to_train_dir, self.image_id_basename)

    def draw_gt_anchor_boxes(self, image: np.ndarray,
                             anchor_labels: np.ndarray,
                             anchor_boxes: np.ndarray,
                             gt_boxes: np.ndarray):
        """
        Draw positive anchor and ground truth boxes on the image.
        
        Args:
            image (numpy.ndarray): A numpy array representing the image.
            anchor_labels (numpy.ndarray): The anchor label denoting whether the anchor box is a positive or
                negative one.
            anchor_boxes (numpy.ndarray): The anchor box coordinates.
            gt_boxes (numpy.ndarray): The ground truth bounding box coordinates.

        Returns:
            An image for which the anchor and ground truth boxes are drawn on to.
            
        """
        thickness = 1
        gt_color = (0, 255, 0)
        anchor_color = (0, 0, 255)

        for k, anchor_box in enumerate(anchor_boxes):
            anchor_label = anchor_labels[k]
            if anchor_label == self.positive_anchor_label:
                image = draw_single_box(image, anchor_box, color=anchor_color, thickness=thickness)

        image = draw_bboxes(image, gt_boxes, color=gt_color, thickness=thickness)

        return image

    def __getitem__(self, index: int):

        # Taking the index-th image.
        image_id = self.image_ids[index]

        # Reading and converting the image into RGB.
        image_file = osp.join(self.path_to_train_dir, self.image_dir_name, "{0}.jpg".format(image_id))
        image = np.array(read_image(image_file))

        # Parsing ground truth boxes.
        annotation_file = osp.join(self.path_to_train_dir, self.annotation_dir_name, "{0}.txt".format(image_id))
        gt_bboxes = parse_annotations(annotation_file)

        # Applying data transformations such as random zoom out, random horizontal flip and so on and so forth.
        image, gt_bboxes = self.data_transform(image, gt_bboxes)

        if not self.is_train:
            return image, gt_bboxes

        # Target transform: Match anchors (default boxes) to ground truth boxes
        target_transforms = self.target_transform(gt_boxes=gt_bboxes.clone(),
                                                  image_size=image.shape[1:3],
                                                  return_anchor_boxes=self.display_anchor_boxes_during_training)

        # uncomment these lines below to check the ground truth and anchor boxes drawn on the image.
        if self.display_anchor_boxes_during_training:

            anchor_boxes = target_transforms[2]

            encoded_gt_labels = target_transforms[1]

            anchor_labels = encoded_gt_labels.clone().cpu().numpy()

            gt_bboxes = gt_bboxes.clone().cpu().numpy()

            copy_image = image.clone()

            copy_image = np.ascontiguousarray(copy_image.permute(1, 2, 0).cpu().numpy())

            debug_img = self.draw_gt_anchor_boxes(image=copy_image,
                                                  anchor_labels=anchor_labels,
                                                  anchor_boxes=anchor_boxes,
                                                  gt_boxes=gt_bboxes)

            path_to_debug_image = os.path.join("text_localization/ctpn/debug")

            if not osp.exists(path_to_debug_image):
                os.makedirs(path_to_debug_image)

            cv2.imwrite(os.path.join(path_to_debug_image, image_id + ".png"), debug_img * 255.0)

        return image, target_transforms[:2]

    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    from text_localization.ctpn.configs import configs

    from functional.utils.dataset import compute_mean_and_std

    dataset = SROIE2019Dataset(configs, is_train=False)

    # Computing the mean and std of the dataset.
    mean, std = compute_mean_and_std(dataset)
    print("mean={0}, std={1}".format(mean, std))
