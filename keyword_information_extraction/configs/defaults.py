import os.path as osp
from yacs.config import CfgNode as CN
from scripts.datasets.dataset_roots import SROIE_ROOT

# It uses yacs which does the job perfectly. For further info,
# check this out: https://github.com/rbgirshick/yacs/blob/master/yacs/config.py

_C = CN()

# ---------------------------------------------------------------------------- #
#                                    Model                                     #
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = "charlm_cnn_highway_lstm"
_C.MODEL.ARGS = [
    ["char_embedding_dim", 15],
    ["char_conv_kernel_sizes", [1, 2, 3, 4, 5, 6]],
    ["char_conv_feature_maps", [25, 50, 75, 100, 125, 150]],  # [25 * kernel_size]
    ["num_highway_layers", 1],
    ["num_lstm_layers", 2],
    ["hidden_size", 300],
    ["dropout", 0.5],
    ["padding_idx", 0]
]

# ---------------------------------------------------------------------------- #
#                                    Loss                                      #
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()
_C.LOSS.IGNORE_INDEX = -1

# ---------------------------------------------------------------------------- #
#                                  Dataset                                     #
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
# The name of the dataset. By default, it is the SROIE2019
_C.DATASET.NAME = "SROIE2019"
# The number of classes for this dataset.
_C.DATASET.NUM_CLASSES = 5
# The train dataset list that specifies the path to folders.
_C.DATASET.TRAIN = [["data-dir", osp.join(SROIE_ROOT, "new-task3-folder")], ["split", "train"]]
# The test dataset list that specifies the path to folders.
_C.DATASET.TEST = [["data-dir", osp.join(SROIE_ROOT, "new-task3-folder")], ["split", "test"]]
# The class names associated to this dataset.
_C.DATASET.LABELS_CLASSES = [["none", 0], ["company", 1], ["date", 2], ["address", 3], ["total", 4]]

# ---------------------------------------------------------------------------- #
#                                  Dataloader                                  #
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.TRAINING = [
    ["batch_size", 10],
    ["shuffle", True],
    ["num_workers", 0],
    ["pin_memory", False],
    ["drop_last", False]
]
_C.DATALOADER.EVALUATION = [
    ["batch_size", 10],
    ["shuffle", False],
    ["num_workers", 0],
    ["pin_memory", False],
    ["drop_last", False]
]

# ---------------------------------------------------------------------------- #
#                                AdamW Solver                                  #
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 30

_C.SOLVER.ADAM = CN()
_C.SOLVER.ADAM.ARGS = [
    ["lr", 1e-3],
    ["eps", 1e-8],
    ["amsgrad", True],
    ["weight_decay", 0.0],
    ["betas", (0.9, 0.999)]
]

_C.SOLVER.SCHEDULER = CN()
_C.SOLVER.SCHEDULER.ARGS = [
    ["gamma", 0.5],
    ["step_size", 5]
]

# ---------------------------------------------------------------------------- #
#                                  Visdom                                      #
# ---------------------------------------------------------------------------- #
_C.VISDOM = CN()
_C.VISDOM.PORT = 8088
_C.VISDOM.ENV_NAME = "Character Aware for NLP: Keyword Information Extraction"

# ---------------------------------------------------------------------------- #
#                               Tabulate grid                                  #
# ---------------------------------------------------------------------------- #
_C.TABULATE = CN()
_C.TABULATE.DATA_LIST = [["Entity", "Recall", "Precision", "F1-score"]]

# The default output dir for all files generated during either training, evaluation or prediction.
_C.OUTPUT_DIR = "keyword_information_extraction/outputs"
