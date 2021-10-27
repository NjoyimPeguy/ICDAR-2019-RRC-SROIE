## Table of contents

1. [Task 1 - Scanned Receipt Text Localisation](#Task1-description)
    1. [Description](#description)
    2. [Evaluation protocol](#eval-protocol)
2. [Why is it a difficult task](#difficult-task)
3. [Crop preprocessing steps](#crop-preprocessing-steps)

## Task 1 - Scanned Receipt Text Localisation <a name="Task1-description"/>

### Description <a name="description"/>

The aim of this task is to accurately localize texts with 4 vertices. The text localization ground truth will be at
least at the level of words.

### Evaluation protocol <a name="eval-protocol"/>

The evaluation of text localisation in this task, the methodology based on DetVal will be implemented. The methodology
address partly the problem of one-to-many and many to one correspondence of detected texts.

## Why is it a difficult task? <a name="difficult-task"/>

This task is difficult since scanned receipts have a lot of variations and are sometimes of low quality.

<div align="center">
    <img src="../scripts/sroie2019/preprocessing/images/X51005361946.jpg" width="250"/>
    <img src="../scripts/sroie2019/preprocessing/images/X51007846370.jpg" width="250"/>
</div>

## Crop preprocessing steps <a name="crop-preprocessing-steps"/>

Before running the training, the [crop preprocessing](../scripts/sroie2019/preprocessing/split_labels.py) (starts at
line 99) method is applied if it was not done the very first time. This method first convert the image to gray scale.
Then gradient is obtained with the Sobel operator which will be used to do the Otsu's binarization. The outline of the
scanned receipt area is decided by morphological transformations with the rectangle kernel.

This step is useful when one wants to remove the extra white space that dominates some scanned receipts and thus the
model will only be focused on the text area.

<div align="center">
    <img src="./figures/crop_preprocessing.png" title="The crop preprocessing" alt="The crop preprocessing"/>
</div>



