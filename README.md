# ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction

> I submitted my results to the competition with the last model checkpoint.
> Unfortunately, my results are not reproducibles accross different machines but were the same on my machine through multiple executions.
> Indeed, after having seeded the training, the results obtained were slightly different on the borrowed NVIDIA GTX 1080.
> This is not surprising because it is mentioned in the Pytorch documentation about [reproducibility](https://pytorch.org/docs/stable/notes/randomness.html).
> That is why I gave up on seeding the training.
> One last important thing to note is this repository aims to be the base code for scientific researches on OCR.

## Table of contents

1. [Highlights](#Highlights)
2. [Challenge](#Challenge)
    1. [Overview](#Overview)
    2. [Dataset and Annotations](#dset-and-annotations)
        1. [Description](#dset-description)
        2. [Mistakes](#dset-mistakes)
        3. [Downloads](#dset-downloads)
3. [Methods](#methods)
4. [Results](#results)
5. [User guide](#user-guide)
    1. [Software & hardware](#soft-hard-ware)
    2. [Conda environment setup](#environment-setup)
    3. [Visualizer](#Visualizer)
6. [Troubleshooting](#troubleshooting)
7. [References](#references)
8. [Citations](#citations)
9. [TODO](#todo)

## Highlights

- **Pytorch 1.7.0**: Currently [Pytorch](https://pytorch.org/) 1.7.0 or higher is supported.
- **Metrics visualization**: Support `Visdom` for real-time loss visualization during training.
- **Automatic mixed precision training**: Train faster
  with [AMP](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) and less GPU memory
  on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores)
- **Pretrained models provided**: Load the pretrained weights.
- **GPU/CPU support for inference**: Runs on GPU or CPU in inference time.
- **Intelligent training procedure**: The state of the model, optimizer, learning rate scheduler and so on... You can
  stop your training and resume training exactly from the checkpoint.

## Challenge

### Overview

Scanned receipts OCR is a process of recognizing text from scanned structured and semi-structured receipts, and invoices
in general. Indeed,

> Scanned receipts OCR and information extraction (SROIE) play critical roles in streamlining document-intensive processes and office automation in many financial, accounting and taxation areas.

For further info, check the [ICDAR19-RRC-SROIE](https://rrc.cvc.uab.es/?ch=13&com=introduction) competition.

### Dataset and Annotations <a name="dset-and-annotations"/>

#### Description <a name="dset-description"/>

The dataset has 1000 whole scanned receipt images. Each receipt image contains around about four key text fields, such
as goods name, unit price and total cost, etc. The text annotated in the dataset mainly consists of digits and English
characters. An example scanned receipt is shown below:

<div align=center><img src="figures/sroie_sample.jpeg" width="300" title="Scanned receipt"/></div>

The dataset is split into a training/validation set (“trainval”), and a test set (“test”). The “trainval” set consists
of 600 receipt images along with their annotations. The “test” set consists of 400 images.

For receipt OCR task, each image in the dataset is annotated with text bounding boxes (bbox) and the transcript of each
text bbox. Locations are annotated as rectangles with four vertices, which are in clockwise order starting from the top.
Annotations for an image are stored in a text file with the same file name. The annotation format is similar to that of
ICDAR2015 dataset, which is shown below:

```
x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1,transcript_1

x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2,transcript_2

x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3,transcript_3

…
```

For the information extraction task, each image in the dataset is annotated with a text file with format shown below:

```json
{
  "company": "STARBUCKS STORE #10208",
  "address": "11302 EUCLID AVENUE, CLEVELAND, OH (216) 229-0749",
  "date": "14/03/2015",
  "total": "4.95"
}
```

#### Mistakes <a name="dset-mistakes"/>

The original dataset provided on the SROIE 2019 competition contains many big mistakes. One of them is the missing file
in `task1_2_test(361p)`. Indeed, the number of files in `task1_2_test(361p)` and `text.task1_2-test（361p)` are not the
same (360 and 361 respectively). The reason is that this filename `X51006619570.jpg` is missing, and it turns out that
it was in `task3-test 347p) -`.

Another mistake lies within the folder `0325updated.task2train(626p)` which is used for
the [Task 3: Keyword Information Extraction](./keyword_information_extraction). Indeed, there are three files for which
the date format is wrong and here are the corrections that were made:

|   Filename   |          Correction           |
| :----------: | :-----------------------------|
| X51005447850 | Turn 20180304 into 04/03/2018 |
| X51005715010 | Turn 25032018 into 25/03/2018 |
| X51006466055 | Turn 20180428 into 28/04/2018 |

#### Downloads <a name="dset-downloads"/>

The **Original dataset** can be found [Google Drive](https://drive.google.com/open?id=1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2)
or [Baidu NetDisk](https://pan.baidu.com/s/1a57eKCSq8SV8Njz8-jO4Ww#list/path=%2FSROIE2019&parentPath=%2F).

Taking into account to what was mentionned in [dataset mistakes](#dset-mistakes) above, you may obviously want to make
the changes by your own, but I have already made the corrections, and it can be downloaded via the bash
script: [sroie2019.sh](scripts/datasets/sroie2019.sh) and here is how to run it:

- Without specifying the directory, then it will download inside the default directory `~/SROIE2019`:

    ```
    sh scripts/datasets/sroie2019.sh
    ```

- Specifying a new directory, let's say `~/dataset/ICDAR2019`:

    ```
    sh scripts/datasets/sroie2019.sh ~/dataset/ICDAR2019
    ```

  Do not forget to specify the new directory inside this [file](scripts/datasets/dataset_roots.py).

For Windows users who do not have bash on their system, you may want to
install [git bash](https://git-scm.com/download/win). Once it is installed, you can set the entire `git bin` folder in
the [environment variables](https://stackoverflow.com/questions/17312348/how-do-i-set-windows-environment-variables-permanently)
.

## Methods <a name="methods"/>

Here are methods used for the competition. Inside each folder representing the task
name, there are documentations of the proposed method and the training, demo and evaluation procedures as well.

- **Task 1 - Text Localization**: Connectionist Text Proposal Network (CTPN).
- **Task 3 - Keyword Information Extraction**: Character-Aware CNN + Highway + BiLSTM (Char LM).

## Results <a name="results"/>

The results are listed as follows (Note that for the task 3, I manually fix each and every OCR mismatches for fair
comparison results):

|  Task  |   Recall  | Precision |  Hmean   | Evaluation Method  |     Model      |   Parameters   | Model Size | Weights |
| :----: | :-------: | :-------: | :------: | :----------------: | :------------: | :------------: | :--------: | :--------------------------------------------------------------------------------------------------------------------------------- |
| Task 1 |   97.16%  |   97.10%  |  97.13%  |      Deteval       |     CTPN       |   16,900,032   |  268.3 MB  | [Last checkpoint]() |
| task 3 |   96.18%  |   97.45%  |  96.81%  |         /          |     Char LM    |   4,740,590    |  75.9 MB   | [last checkpoint]() |

## User guide <a name="user-guide"/>

### Hardware <a name="soft-hard-ware"/>

NVIDIA GPU (with tensor cores in order to use automatic mixed precision) + CUDNN if possible is strongly recommended.
It's also possible to run the program on CPU only, but it will be extremely slow.

Besides, all the experiments and results were performed on my personal gaming computer:

- Alienware Area-51m R2
- 10th Gen Intel(R) Core(TM) i9 10900K (10-Core, 20 MB Cache, 3.7GHz to 5.3GHz w/Thermal Velocity Boost)
- NVIDIA® GeForce® RTX 2070 SUPER™, 8Go GDDR6
- OS: Dual boot Windows/Ubuntu 20.04

### Conda environment setup <a name="environment-setup"/>

For Mac, Windows and Linux users, if `conda` is not installed, then you need to follow
this [documentation](https://docs.continuum.io/anaconda/install/).

1. Updating conda

   Before installation, we need to make sure `conda` is updated.

   ```
   conda update conda
   ```

2. Creating an environment from a file

   ```
   conda env create -f env/environment.yml
   ```

   This will create a new conda environment named `SROIE2019` on your system, which will give you all the packages
   needed for this repo.

3. Activating the new environment

   ```
   conda activate SROIE2019
   ```

   or

   ```
   source activate SROIE2019
   ```

   If you want to deactivate the environment, you can simply do: `conda deactivate`

4. Verify that the new environment was installed correctly

   ```
   conda env list
   ```

   or

   ```
   conda info --envs
   ```

for further info, you can check
the [manager](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
.

### Visualizer

To use vidsom, you must make sure the server is running before you run the training.

#### Starting the server with

```
python3 -m visdom.server
```

or simply

```
visdom
```

#### Visdom interface

In your browser, you can go to:

http://localhost:8097

You will see the visdom interface:

<div align="center"><img src="figures/visdom-main.png" alt="visdom" width="460"/></div>

One important thing to remember...You can launch the server with a specific port. Let's say: `visdom --port 8198`. But
you need to make sure the [Visualizer](text_localization/ctpn/utils/misc.py) runs with the port `8198`.

For further info on Visdom, you can check this: https://github.com/fossasia/visdom

## Troubleshooting <a name="troubleshooting"/>

If you have issues running or compiling this code, there are a list of common issues
in [TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, then feel free to open a new issue.

## References <a name="references"/>

This repository is influenced by great works such as :

- [Luffic-SSD](https://github.com/lufficc/SSD) for the data augmentation and anchor matching parts.
- [eragonruan](https://github.com/eragonruan/text-detection-ctpn), [courao](https://github.com/courao/ocr.pytorch)
  , [tranbahien](https://github.com/tranbahien/CTPN-TensorFlow) for the implementation of the CTPN used to tackle the
  text localization.
- [eadst](https://github.com/eadst/CEIR) for the implementation for removing the extra white space in the scanned
  receipts for the text localization.
- [HephaestusProject](https://github.com/HephaestusProject/pytorch-CharLM) for the implementation of Character-Aware
  Neural Language Models used to tackle the keyword-information extraction.

## Citations <a name="citations"/>

If you use this project in your research, please cite it as follows:

```text
@misc{blackstar1313_sroie_2019,
  author = {Njoyim Tchoubith Peguy Calusha},
  title  = {ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction},
  year   = {2021},
  url    = {https://github.com/BlackStar1313/ICDAR-2019-RRC-SROIE}
}
```


## TODO <a name="todo"/>

Here is a to-do list which should be complete subsequently.

* [ ] Support for the Docker images.
* [ ] Support for the **task 2: Scanned Receipts OCR**.
