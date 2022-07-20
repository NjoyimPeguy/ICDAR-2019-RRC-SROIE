import os
import sys

sys.path.append(os.getcwd())

import cv2
import shutil
import numpy as np
import os.path as osp

from functional.utils.box import draw_bboxes
from functional.utils.dataset import read_image, make_directories, parse_annotations
from functional.data.transformation.computer_vision import ToSobelGradient, ToMorphology, CropImage, ConvertColor


def generate_multiple_directories(*dir_paths):
    for path_to_dir in dir_paths:
        make_directories(path_to_dir)


def generate_txt_name(path_to_image, path_to_txt_name, name: str = "trainval.txt"):
    image_files = np.array(list(sorted(os.scandir(path=path_to_image), key=lambda f: f.name)))
    trainval_file = osp.join(path_to_txt_name, name)
    write_to(trainval_file, image_files)


def get_basename(file):
    return osp.splitext(file.name)[0]


def write_to(file, image_files):
    with open(file, encoding='utf-8', mode='w') as file:
        for jpg_file in image_files:
            image_id = osp.splitext(jpg_file.name)[0]
            file.write("{0}\n".format(image_id))


def adjust_label(annotation_path: str, cropped_pixels_width: float, cropped_pixels_height: float):
    """
    Calculate the new bounding box coordinates after an image was cropped.
    For instance, if 147 pixels were removed on the x-axis and 25 pixels on the y-axis,
    then we need to subtract 147 on the bounding box x-coordinates and 25 on the y-coordinates.
    For further info, check this out:
    https://stackoverflow.com/questions/59652250/after-cropping-a-image-how-to-find-new-bounding-box-coordinates

    Args:
        annotation_path: The path to the annotations in .txt format.
        cropped_pixels_width: The number of pixels that were removed on the x-axis.
        cropped_pixels_height: The number of pixels that were removed on the y-axis.

    Returns:
        The new annotations for an image that was cropped.

    """
    new_lines = ""
    with open(annotation_path, encoding='utf-8', mode='r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip('\r\n').split(',')
            coordinates = np.array(list(map(np.float32, parts[:8])))
            coordinates[0::2] -= cropped_pixels_width
            coordinates[1::2] -= cropped_pixels_height
            new_lines += ",".join(str(round(coord)) for coord in coordinates)
            for word in parts[8:]:
                new_lines += ','
                new_lines += word
            new_lines += "\r\n"
    return new_lines


def crop_preprocessing(source_folder: str,
                       parent_folder: str,
                       image_dir_name: str,
                       annotation_dir_name: str,
                       image_id_names: str):
    print("Fixing scale variation resolution before training...\n")

    new_image_path = osp.join(parent_folder, image_dir_name)

    new_annotation_path = osp.join(parent_folder, annotation_dir_name)

    generate_multiple_directories(new_image_path, new_annotation_path, parent_folder)

    original_files = np.array(list(sorted(os.scandir(path=source_folder), key=lambda file: file.name)))

    jpg_files = [get_basename(file) for file in original_files if file.name.endswith("jpg")]

    txt_files = [get_basename(file) for file in original_files if file.name.endswith("txt")]

    # We only want a unique combination of image/annotation files
    image_ids = sorted(list(set(jpg_files).intersection(txt_files)))

    # A set of classes for removing extra white space
    crop_image = CropImage()
    to_gray_color = ConvertColor()
    to_morphology = ToMorphology()
    thresh_and_blur = ToSobelGradient(cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for i, image_id in enumerate(image_ids):
        image_name = image_id + ".jpg"
        image_path = osp.join(source_folder, image_name)
        image = np.array(read_image(image_path))
        annotation_name = image_id + ".txt"
        old_annotation_path = osp.join(source_folder, annotation_name)
        # One can change this threshold value and see the results
        if image.shape[1] <= 990:
            shutil.copyfile(image_path, osp.join(new_image_path, image_name))
            shutil.copyfile(old_annotation_path, osp.join(new_annotation_path, annotation_name))
            process_type = "Copying image"
        else:
            process_type = "Cropping image"
            original_image = image.copy()
            gray_image = to_gray_color(image)[0]
            threshed_image = thresh_and_blur(gray_image)
            erode_iterations = 6
            dilate_iterations = 6
            if image_id in ("X51005442378", "X51005442378(1)"):  # Had troubles with these two images.
                erode_iterations = 1
                dilate_iterations = 1
            morpho_image = to_morphology(threshed_image,
                                         erode_iterations=erode_iterations,
                                         dilate_iterations=dilate_iterations)
            cropped_image, cropped_pixels_width, cropped_pixels_height = crop_image(morpho_image, original_image)

            if cropped_image.size == 0:
                raise ValueError("The cropped image is empty!!")

            cv2.imwrite(osp.join(new_image_path, image_name), cropped_image)

            new_lines = adjust_label(old_annotation_path, cropped_pixels_width, cropped_pixels_height)
            with open(osp.join(new_annotation_path, annotation_name), encoding='utf-8', mode='w') as file:
                file.write(new_lines)

        print("{0}/{1}: {2} '{3}'\n".format(i + 1, len(image_ids), process_type, image_name))

    generate_txt_name(new_image_path, parent_folder, name=image_id_names)


def crop_preprocess_step_by_step(adjust_labels: bool = False):
    print("Scale variation resolution step-by-step...\n")

    # ---------------------------------------------------------------------------- #
    #                     SROIE 2019 preprocessing first phase                     #
    # ---------------------------------------------------------------------------- #
    path_to_image = osp.normpath("text_localization/demo/images/X51005442388.jpg")

    annotation_path = osp.join("text_localization/demo/annotations/X51005442388.txt")

    filename = osp.splitext(osp.basename(path_to_image))[0]

    image = np.array(read_image(path_to_image))

    original_image = image.copy()

    gray_image = ConvertColor(current="RGB", transform="GRAY")(image)[0]

    threshed_image = ToSobelGradient(cv2.THRESH_BINARY + cv2.THRESH_OTSU)(gray_image)  # Otsu's binarization

    morpho_image = ToMorphology()(threshed_image, erode_iters=6, dilate_iters=6)  # Morphology Ex

    output = CropImage(draw_contours=True)(morpho_image, original_image)  # Find contours

    cropped_image, image_with_contours, cropped_pixels_width, cropped_pixels_height = output

    # Creating the directory of this preprocess
    save_path = osp.normpath("text_localization/demo/crop_preprocessing_results")
    if not osp.exists(save_path) or not osp.isdir(save_path):
        os.makedirs(save_path)

    # Creating a dict where the keys are the name of to preprocess and values are the preprocessed image.
    preprocessed_images = {
        "original_image": image,
        "Otsu's_binarization_step1": threshed_image,
        "MorphoEx_step2": morpho_image,
        "Contours_step3": image_with_contours,
        "Crop_step4": cropped_image
    }

    # Saving the preprocessed figures
    image_name, image_ext = osp.splitext(osp.basename(path_to_image))
    for preprocess_name, preprocessed_image in preprocessed_images.items():
        path = osp.join(save_path, image_name + "_" + preprocess_name + image_ext)
        is_written = cv2.imwrite(path, preprocessed_image)
        if not is_written:
            raise ValueError("An error has occurred when saving the path_to_file! Check the syntax of your filename.")

    if adjust_labels:
        new_lines = adjust_label(annotation_path, cropped_pixels_width, cropped_pixels_height)
        new_annotation_path = osp.join(save_path, filename + "_adjusted.txt")
        with open(osp.join(new_annotation_path), encoding='utf-8', mode='w') as file:
            file.write(new_lines)

        bboxes = parse_annotations(annotation_path=new_annotation_path)

        drawn_image = draw_bboxes(image=cropped_image, bboxes=bboxes)

        cv2.namedWindow("Image bounding boxes", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Image bounding boxes", drawn_image)
        cv2.resizeWindow("Image bounding boxes", 1000, 950)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    crop_preprocess_step_by_step(adjust_labels=True)
