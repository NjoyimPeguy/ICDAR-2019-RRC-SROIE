import os
import cv2
import sys
import shutil
import numpy as np
import os.path as osp

sys.path.append(os.getcwd())

from text_localization.ctpn.utils.dset import read_image, make_directories
from text_localization.ctpn.data.preprocessing.transformations import ToSobelGradient, ToMorphology, \
    CropImage, ConvertColor


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


def adjust_label(old_annotation_path, cropped_pixels_width, cropped_pixels_height):
    """
    Calculate the new bounding box coordinates after an image was cropped.
    For instance, if 147 pixels were removed on the x-axis and 25 pixels on the y-axis,
    then we need to subtract 147 on the bounding box x-coordinates and 25 on the y-coordinates.
    For further info, check this out: https://stackoverflow.com/questions/59652250/after-cropping-a-image-how-to-find-new-bounding-box-coordinates
    
    Args:
        old_annotation_path:    The path to the old annotations.
        cropped_pixels_width:   The number of pixels that were removed on the x-axis.
        cropped_pixels_height:  The number of pixels that were removed on the y-axis.
        
    Returns:
        The new annotations for an image that was cropped.
        
    """
    new_lines = ""
    with open(old_annotation_path, encoding='utf-8', mode='r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip('\r\n').split(',')
            coords = np.array(list(map(np.float32, parts[:8])))
            coords[0::2] -= cropped_pixels_width
            coords[1::2] -= cropped_pixels_height
            new_lines += ",".join(str(round(coord)) for coord in coords)
            for word in parts[8:]:
                new_lines += ','
                new_lines += word
            new_lines += "\r\n"
    return new_lines


def generate_corrected_base_dataset(source_folder: str,
                                    parent_folder: str,
                                    image_dir_name: str,
                                    annotation_dir_name: str,
                                    image_id_names: str):
    print("Generating the corrected base dataset...\n")
    
    new_image_path = osp.join(parent_folder, image_dir_name)
    new_annotation_path = osp.join(parent_folder, annotation_dir_name)
    
    generate_multiple_directories(new_image_path, new_annotation_path, parent_folder)
    
    original_files = np.array(list(sorted(os.scandir(path=source_folder), key=lambda file: file.name)))
    jpg_files = [get_basename(file) for file in original_files if file.name.endswith("jpg")]
    txt_files = [get_basename(file) for file in original_files if file.name.endswith("txt")]
    
    # We only want a unique combination of image/annotation files
    final_train_image_ids = sorted(list(set(jpg_files).intersection(txt_files)))
    
    for i, train_image_id in enumerate(final_train_image_ids):
        image_name = train_image_id + ".jpg"
        image_path = osp.join(source_folder, image_name)
        annotation_name = train_image_id + ".txt"
        annotation_path = osp.join(source_folder, annotation_name)
        shutil.copyfile(image_path, osp.join(new_image_path, image_name))
        shutil.copyfile(annotation_path, osp.join(new_annotation_path, annotation_name))
    
    generate_txt_name(new_image_path, parent_folder, name=image_id_names)


def crop_preprocessing(source_folder: str,
                       parent_folder: str,
                       image_dir_name: str,
                       annotation_dir_name: str,
                       image_id_names: str):
    print("\nFixing scale variation resolution before running the training...\n")
    
    new_image_path = osp.join(parent_folder, image_dir_name)
    new_annotation_path = osp.join(parent_folder, annotation_dir_name)
    
    generate_multiple_directories(new_image_path, new_annotation_path, parent_folder)
    
    final_train_image_ids = []
    with open(osp.join(source_folder, image_id_names), mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            final_train_image_ids.append(line.strip())
    
    # A set of classes for removing extra white space
    to_gray_color = ConvertColor(current="RGB", transform="GRAY")
    thresh_and_blur = ToSobelGradient(cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    to_morphology = ToMorphology()
    crop_image = CropImage()
    
    for i, train_image_id in enumerate(final_train_image_ids):
        image_name = train_image_id + ".jpg"
        image_path = osp.join(source_folder, image_dir_name, image_name)
        image = np.array(read_image(image_path))
        annotation_name = train_image_id + ".txt"
        annotation_path = osp.join(source_folder, annotation_dir_name, annotation_name)
        # One can change this threshold value and see the results
        if image.shape[1] <= 990:
            shutil.copyfile(image_path, osp.join(new_image_path, image_name))
            shutil.copyfile(annotation_path, osp.join(new_annotation_path, annotation_name))
            process_type = "Copying image"
        else:
            process_type = "Cropping image"
            original_image = image.copy()
            gray_image = to_gray_color(image)[0]
            threshed_image = thresh_and_blur(gray_image)
            morpho_image = to_morphology(threshed_image)
            cropped_image, cropped_pixels_width, cropped_pixels_height = crop_image(morpho_image, original_image)
            if cropped_image.size == 0:
                raise ValueError("The cropped image is empty!!")
            cv2.imwrite(osp.join(new_image_path, image_name), cropped_image)
            new_lines = adjust_label(annotation_path, cropped_pixels_width, cropped_pixels_height)
            with open(osp.join(new_annotation_path, annotation_name), encoding='utf-8', mode='w') as file:
                file.write(new_lines)
        print("{0}/{1}: {2} '{3}'\n".format(i + 1, len(final_train_image_ids), process_type, image_name))
    
    generate_txt_name(new_image_path, parent_folder, name=image_id_names)


def crop_preprocess_step_by_step(adjust_labels: bool = False):
    print("\nScale variation resolution step-by-step...\n")
    
    # ---------------------------------------------------------------------------- #
    #                     SROIE 2019 preprocessing first phase                     #
    # ---------------------------------------------------------------------------- #
    path_to_image = osp.normpath("scripts/sroie2019/preprocessing/images/X51007846370.jpg")
    filename = osp.splitext(osp.basename(path_to_image))[0]
    image = np.array(read_image(path_to_image))
    original_image = image.copy()
    
    gray_image = ConvertColor(current="RGB", transform="GRAY")(image)[0]
    
    threshed_image = ToSobelGradient(cv2.THRESH_BINARY + cv2.THRESH_OTSU)(gray_image)  # Otsu's binarization
    
    morpho_image = ToMorphology()(threshed_image)  # Morphology Ex
    
    cropped_image, image_with_contours, cropped_pixels_width, cropped_pixels_height = CropImage(draw_contours=True)(
        morpho_image, original_image)  # Find contours
    
    if adjust_labels:
        annotation_path = osp.join("scripts/sroie2019/preprocessing/annotations", filename + ".txt")
        new_lines = adjust_label(annotation_path, cropped_pixels_width, cropped_pixels_height)
        new_annotation_path = osp.join("scripts/sroie2019/preprocessing/annotations", filename + "_adjusted.txt")
        with open(osp.join(new_annotation_path), encoding='utf-8', mode='w') as file:
            file.write(new_lines)
    
    # Creating the directory of this preprocess
    save_path = osp.normpath("scripts/sroie2019/preprocessing/results/")
    if not osp.exists(save_path) or not osp.isdir(save_path):
        os.makedirs(save_path)
    
    # Creating a dict where the keys are the name of the preprocess and values are the preprocessed image.
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
            raise ValueError("An error has occurred when saving the file! Check the syntax of your filename.")


if __name__ == '__main__':
    crop_preprocess_step_by_step(adjust_labels=True)
