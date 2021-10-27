import os
import cv2
import sys
import copy
import numpy as np
import os.path as osp

sys.path.append(os.getcwd())

from text_localization.ctpn.data.postprocessing import nms


if __name__ == "__main__":
    
    # construct a list containing the nms_images_test that will be examined
    # along with their respective bounding boxes
    # and random confidence scores.
    images = [
        (osp.normpath("text_localization/ctpn/data/postprocessing/test/images/audrey.jpg"), np.array([
            (12, 84, 140, 212),
            (24, 84, 152, 212),
            (36, 84, 164, 212),
            (12, 96, 140, 224),
            (24, 96, 152, 224),
            (24, 108, 152, 236),
            (32, 84, 120, 202),
            (24, 74, 152, 222),
            (16, 84, 134, 212),
            (12, 96, 140, 214),
            (24, 76, 152, 224),
            (34, 118, 142, 246)]),
         np.array([0.71553708, 0.44149134, 0.56920083, 0.66437074, 0.94646953,
                   0.5710134, 0.59851521, 0.86266735, 0.35275677, 0.63534861,
                   0.92070096, 0.58120545])),
        # np.random.uniform(0.3, 0.98, (12,))),
        
        (osp.normpath("text_localization/ctpn/data/postprocessing/test/images/Musician.jpg"), np.array([
            (114, 60, 178, 124),
            (120, 60, 184, 124),
            (114, 66, 178, 130)]),
         np.array([0.39508096, 0.30129297, 0.55505935])),
        # np.random.uniform(0.3, 0.98, (3,))),
        
        (osp.normpath("text_localization/ctpn/data/postprocessing/test/images/gpripe.jpg"), np.array([
            (12, 30, 76, 94),
            (12, 36, 76, 100),
            (72, 36, 200, 164),
            (84, 48, 212, 176)]),
         np.array([0.60069897, 0.71678238, 0.59429882, 0.84540743]))
        # np.random.uniform(0.3, 0.98, (4,)))
    ]
    
    # loop over the nms_images_test
    for i, (imagePath, boundingBoxes, confidence_scores) in enumerate(images):
        # load the image and clone it
        # print ("[x] %d initial bounding boxes" % (len(boundingBoxes)))
        image_name, image_ext = osp.splitext(osp.basename(imagePath))
        original_img = cv2.imread(imagePath)
        copied_img = copy.deepcopy(original_img)
        
        # loop over the bounding boxes for each image and draw them
        for (startX, startY, endX, endY) in boundingBoxes:
            cv2.rectangle(original_img, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        picked_indices = nms(boundingBoxes, confidence_scores, 0.5)
        
        # loop over the picked bounding boxes and draw them
        for (startX, startY, endX, endY) in boundingBoxes[picked_indices]:
            cv2.rectangle(copied_img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # # save the nms_images_test
        path_to_saved_images = osp.normpath(
            "text_localization/ctpn/data/postprocessing/test/images/" + image_name + "_nms" + image_ext)
        cv2.imwrite(path_to_saved_images, copied_img)