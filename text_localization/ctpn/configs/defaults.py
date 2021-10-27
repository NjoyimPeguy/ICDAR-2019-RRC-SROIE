from yacs.config import CfgNode as CN

# It uses yacs which does the job perfectly. For further info,
# check this out: https://github.com/rbgirshick/yacs/blob/master/yacs/config.py

_C = CN()

# ---------------------------------------------------------------------------- #
#                                  CTPN                                        #
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARGS = [
    ["f_out_channels", 256],
    ["n_classes", 2],
    ["n_anchors", 10],
    ["pretrained_backbone", True]
]

# ---------------------------------------------------------------------------- #
#                                   RPN                                        #
# ---------------------------------------------------------------------------- #
_C.RPN = CN()
_C.RPN.POSITIVE_JACCARD_OVERLAP = 0.5
_C.RPN.NEGATIVE_JACCARD_OVERLAP = 0.3

# ---------------------------------------------------------------------------- #
#                                Anchor boxes                                  #
# ---------------------------------------------------------------------------- #
_C.ANCHOR = CN()
_C.ANCHOR.SCALE = 16.0
_C.ANCHOR.POSITIVE_LABEL = 1
_C.ANCHOR.NEGATIVE_LABEL = 0
_C.ANCHOR.IGNORE_LABEL = -1
_C.ANCHOR.HEIGHTS = [11, 15, 22, 32, 45, 65, 93, 133, 190, 273]

# ---------------------------------------------------------------------------- #
#                                    TEXT LINE                                 #
# ---------------------------------------------------------------------------- #
_C.TEXTLINE = CN()
_C.TEXTLINE.MIN_SIZE_SIM = 0.7
_C.TEXTLINE.MIN_V_OVERLAPS = 0.7
_C.TEXTLINE.MAX_HORIZONTAL_GAP = 20

# ---------------------------------------------------------------------------- #
#                                 TEXT PROPOSALS                               #
# ---------------------------------------------------------------------------- #
_C.TEXT_PROPOSALS = CN()
_C.TEXT_PROPOSALS.MIN_SCORE = 0.5
_C.TEXT_PROPOSALS.NMS_THRESH = 0.1

# ---------------------------------------------------------------------------- #
#                                  Input                                       #
# ---------------------------------------------------------------------------- #
_C.IMAGE = CN()
_C.IMAGE.FEAT_STRIDE = 16
_C.IMAGE.SIZE = [1024, 2048]

# This is the mean and std for the SROIE dataset.
# It is not recommended to use the mean and std form a given pretrained model from Pytorch (https://pytorch.org/vision/stable/models.html)
# because it was trained over the ImageNet dataset which is a dataset completely different from the SROIE.
# Another reason to change the mean and std is we are doing finetuning instead of feature extraction.
# For further info, check out the difference: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
_C.IMAGE.PIXEL_STD = [0.20720498, 0.20747886, 0.20765]
_C.IMAGE.PIXEL_MEAN = [0.92088137, 0.92047861, 0.92000766]

# ---------------------------------------------------------------------------- #
#                                  Dataloader                                  #
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.ARGS = [
    ["batch_size", 1],
    ["num_workers", 4],
    ["shuffle", True],
    ["pin_memory", True],
    ["drop_last", False],
    ["multiprocessing_context", "spawn"]
]

# ---------------------------------------------------------------------------- #
#                                  AdamW Solver                                #
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.LR = 1e-4
_C.SOLVER.EPS = 1e-8
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.AMSGRAD = True
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.BETAS = [0.9, 0.999]
_C.SOLVER.MAX_ITERATIONS = 60000
_C.SOLVER.LR_DECAY_STEPS = [20000, 40000]

# ---------------------------------------------------------------------------- #
#                                  Visdom                                      #
# ---------------------------------------------------------------------------- #
_C.VISDOM = CN()
_C.VISDOM.PORT = 8088
_C.VISDOM.ENV_NAME = "CTPN: Text Localization"

# The output dir for all files generated during either training, evaluation or prediction.
_C.OUTPUT_DIR = "text_localization/ctpn/outputs/"
