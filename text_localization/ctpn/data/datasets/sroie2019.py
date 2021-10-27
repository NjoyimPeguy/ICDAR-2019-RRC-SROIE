import os
import cv2
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from scripts.datasets.dataset_roots import SROIE_ROOT
from text_localization.ctpn.data.preprocessing.transformations import TargetTransform
from text_localization.ctpn.data.preprocessing.augmentation import TrainDataTransformation
from text_localization.ctpn.utils.dset import read_image, read_image_ids, parse_annotations
from scripts.sroie2019.preprocessing import generate_corrected_base_dataset, crop_preprocessing


class SROIE2019Dataset(Dataset):
    
    def __init__(self, configs: dict, is_train: bool = True):
        """
        An abstract representation of the SROIE 2019 dataset.
        
        Args:
            configs: The config file.
            is_train: A boolean indicating whether information about image and boxes comes from training or test.
            
        """
        super(SROIE2019Dataset, self).__init__()
        
        self.is_train = is_train
        
        self.image_id_basename = "trainval.txt"
        self.image_dir_name = "JPEGImages"
        self.annotation_dir_name = "Annotations"
        
        source_folder = osp.join(SROIE_ROOT, "0325updated.task1train(626p)")
        if not osp.exists(source_folder):
            raise ValueError("You have completely forgotten to download the main dataset!")
        
        path_to_base_dir = osp.join(SROIE_ROOT, "new-task1-folder")
        if not osp.exists(path_to_base_dir):
            os.makedirs(path_to_base_dir)
        
        path_to_base_dataset = osp.join(path_to_base_dir, "base")
        if not osp.exists(path_to_base_dataset):
            os.makedirs(path_to_base_dataset)
        if len(os.listdir(path_to_base_dataset)) == 0:
            generate_corrected_base_dataset(source_folder=source_folder,
                                            parent_folder=path_to_base_dataset,
                                            image_dir_name=self.image_dir_name,
                                            annotation_dir_name=self.annotation_dir_name,
                                            image_id_names=self.image_id_basename)
        
        self.path_to_train_dir = osp.join(path_to_base_dir, "crop")
        if not osp.exists(self.path_to_train_dir):
            os.makedirs(self.path_to_train_dir)
        if len(os.listdir(self.path_to_train_dir)) == 0:
            crop_preprocessing(source_folder=path_to_base_dataset,
                               parent_folder=self.path_to_train_dir,
                               image_dir_name=self.image_dir_name,
                               annotation_dir_name=self.annotation_dir_name,
                               image_id_names=self.image_id_basename)
        
        self.data_transform = TrainDataTransformation(configs)
        
        self.target_transform = TargetTransform(configs)
        
        self.image_ids = read_image_ids(self.path_to_train_dir, self.image_id_basename)
        
        self.positive_anchor_label = configs.ANCHOR.POSITIVE_LABEL
    
    def draw_boxes(self, image: np.ndarray,
                   anchor_labels: np.ndarray,
                   anchor_boxes: np.ndarray,
                   gt_boxes: np.ndarray):
        """
        Draw positive anchor and ground truth boxes on the image.
        
        Args:
            image: A numpy array representing the image.
            anchor_labels: The anchor label denoting whether the anchor box is a positive or negative one.
            anchor_boxes: The anchor box coordinates.
            gt_boxes: The ground truth bounding box coordinates.

        Returns:
            An image for which the anchor and ground truth boxes are drawn on to.
            
        """
        thickness = 1
        gt_color = (0, 255, 0)
        anchor_color = (0, 0, 255)
        
        for k, anchor_box in enumerate(anchor_boxes):
            anchor_label = anchor_labels[k]
            xmin, ymin, xmax, ymax = anchor_box
            if anchor_label == self.positive_anchor_label:
                pt1 = (int(xmin), int(ymin))
                pt2 = (int(xmax), int(ymax))
                image = cv2.rectangle(image, pt1, pt2, color=anchor_color, thickness=thickness)
        
        for box in gt_boxes:
            xmin, ymin, xmax, ymax = box
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            image = cv2.rectangle(image, pt1, pt2, color=gt_color, thickness=thickness)
        
        return image
    
    def __getitem__(self, index):
        
        # Taking the index-th image.
        image_id = self.image_ids[index]
        
        # Reading and converting the image into RGB.
        image_file = osp.join(self.path_to_train_dir, self.image_dir_name, "{0}.jpg".format(image_id))
        image = np.array(read_image(image_file))
        
        # Parsing ground truth boxes.
        annotation_file = osp.join(self.path_to_train_dir, self.annotation_dir_name, "{0}.txt".format(image_id))
        gt_boxes = parse_annotations(annotation_file)
        
        if not self.is_train:
            return image, gt_boxes
        
        # Applying data transformations such as random zoom out, random horizontal flip and so on and so forth.
        image, gt_boxes = self.data_transform(image, gt_boxes)
        
        # Target transform: Match anchors (default boxes) to ground truth boxes
        targets, anchor_boxes = self.target_transform(gt_boxes=gt_boxes, image_size=image.shape[1:3])
        
        # uncomment these lines below to check the ground truth and anchor boxes drawn on the image.
        # copy_img = image.detach().clone().permute(1, 2, 0).cpu().numpy()
        # copy_img = np.ascontiguousarray(copy_img)
        # _, encoded_gt_labels = targets
        # anchor_labels = encoded_gt_labels.detach().clone().cpu().numpy()
        # gt_bboxes = gt_boxes.detach().clone().cpu().numpy()
        # debug_img = self.draw_boxes(image=copy_img * 255.,
        #                             anchor_labels=anchor_labels,
        #                             anchor_boxes=anchor_boxes,
        #                             gt_boxes=gt_bboxes)
        # path_to_debug_image = os.path.join("text_localization/ctpn/debug")
        # if not osp.exists(path_to_debug_image):
        #     os.makedirs(path_to_debug_image)
        # cv2.imwrite(os.path.join(path_to_debug_image, image_id + ".png"), debug_img)
        
        return image, targets
    
    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    from text_localization.ctpn.configs import configs
    from text_localization.ctpn.utils.dset import compute_mean_and_std
    
    dataset = SROIE2019Dataset(configs)
    
    # Computing the mean and std of the dataset
    mean, std = compute_mean_and_std(dataset)
    print("mean={0}, std={1}".format(mean / 255.0, std / 255.0))
