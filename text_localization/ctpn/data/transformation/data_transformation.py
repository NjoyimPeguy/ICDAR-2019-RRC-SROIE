import os
import sys

sys.path.append(os.getcwd())

import cv2
import numpy as np

from functional.data.transformation.computer_vision.base import AbstractTransformation


class SplitBBoxes(AbstractTransformation):

    def __init__(self, anchor_shift: int):
        """
        Split the ground truth bounding boxes into a sequence of fine-scale text proposals,

        where each proposal generally represents a small part of a text line, e.g., a text piece with 16-pixel width.
        
        Args:
            anchor_shift (int): The width of each split box.
            
        """

        self.anchor_shift: int = anchor_shift

    def apply(self, image: np.ndarray, gt_bboxes: np.ndarray):
        # Now we split bounding box coordinates according to the anchor shift value.
        new_gt_bboxes = []

        for i, bbox in enumerate(gt_bboxes):
            xmin, ymin, xmax, ymax = bbox

            bbox_ids = np.arange(int(np.floor(1.0 * xmin / self.anchor_shift)),
                                 int(np.ceil(1.0 * xmax / self.anchor_shift)))

            new_bboxes = np.zeros(shape=(len(bbox_ids), 4))

            new_bboxes[:, 0] = bbox_ids * self.anchor_shift

            new_bboxes[:, 1] = ymin

            new_bboxes[:, 2] = (bbox_ids + 1.0) * self.anchor_shift

            new_bboxes[:, 3] = ymax

            new_gt_bboxes.append(new_bboxes)

        # Bounding boxes must be within the image size.
        new_gt_bboxes = np.concatenate(new_gt_bboxes, axis=0)

        return image, new_gt_bboxes


if __name__ == "__main__":
    import os.path as osp

    from functional.utils.box import draw_bboxes

    from functional.data.transformation.computer_vision import Resize

    from functional.utils.dataset import read_image, parse_annotations

    path = osp.normpath("text_localization/demo/images/X00016469671.jpg")

    filename = osp.basename(path)

    original_image = np.array(read_image(path))

    annotation_path = osp.normpath("text_localization/demo/annotations/X00016469671.txt")

    bboxes = parse_annotations(annotation_path)

    original_image, bboxes = Resize(image_size=(1024, 2048), resize_bboxes=True)(original_image, bboxes)

    original_image, bboxes = SplitBBoxes(anchor_shift=16)(original_image, bboxes)

    drawn_image = draw_bboxes(image=original_image, bboxes=bboxes)

    cv2.namedWindow("Image with split bounding boxes", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Image with split bounding boxes", drawn_image)
    cv2.resizeWindow("Image with split bounding boxes", 1000, 950)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
