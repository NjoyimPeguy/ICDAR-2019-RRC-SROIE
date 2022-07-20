from yacs.config import CfgNode as CN

# It uses yacs which does the job perfectly. For further info,
# check this out: https://github.com/rbgirshick/yacs/blob/master/yacs/config.py

_C = CN()

# ---------------------------------------------------------------------------- #
#                                  CTPN                                        #
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARGS = [
    ["n_classes", 2],
    ["n_anchors", 10],
    ["hidden_size", 128],
    ["f_out_channels", 512],
    ["pretrained_backbone", True]
]

# ---------------------------------------------------------------------------- #
#                               Loss functions                                 #
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()
_C.LOSS.LAMBDA_REG = 2.0
_C.LOSS.LAMBDA_CLS = 1.0
_C.LOSS.NEG_POS_RATIO = 3.0

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
_C.ANCHOR.SHIFT = 16
_C.ANCHOR.IGNORE_LABEL = -1
_C.ANCHOR.POSITIVE_LABEL = 1
_C.ANCHOR.NEGATIVE_LABEL = 0
_C.ANCHOR.DISPLAY = False
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
_C.TEXT_PROPOSALS.MIN_SCORE = 0.9
_C.TEXT_PROPOSALS.NMS_THRESH = 0.3

# ---------------------------------------------------------------------------- #
#                                  Input                                       #
# ---------------------------------------------------------------------------- #
_C.IMAGE = CN()
_C.IMAGE.FEAT_STRIDE = 16
_C.IMAGE.SIZE = [1024, 2048]
_C.IMAGE.PIXEL_STD = [0.20037157, 0.18366718, 0.19631825]
_C.IMAGE.PIXEL_MEAN = [0.90890862, 0.91631571, 0.90724233]

# ---------------------------------------------------------------------------- #
#                                  Dataloader                                  #
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.ARGS = [
    ["batch_size", 1],
    ["num_workers", 8],
    ["shuffle", True],
    ["pin_memory", True],
    ["drop_last", False],
    ["persistent_workers", True]
]

# ---------------------------------------------------------------------------- #
#                                       Solver                                 #
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 90

_C.SOLVER.ADAM = CN()
_C.SOLVER.ADAM.ARGS = [
    ["lr", 1e-4],
    ["eps", 1e-8],
    ["amsgrad", True],
    ["weight_decay", 0.0],
    ["betas", (0.9, 0.999)]
]

# ---------------------------------------------------------------------------- #
#                                  Visdom                                      #
# ---------------------------------------------------------------------------- #
_C.VISDOM = CN()
_C.VISDOM.PORT = 8088
_C.VISDOM.ENV_NAME = "CTPN: Text Localization"

# The output dir for all files generated during either training, evaluation or prediction.
_C.OUTPUT_DIR = "text_localization/ctpn/outputs/"
