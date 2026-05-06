#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False
_C.OUTPUT_DIR = "./output"
_C.RUN_N_TIMES = 5
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = None

# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.TRANSFER_TYPE = "linear"  # linear, end2end, prompt, prompt_vk, prompt_fourier, adapter, ...
_C.MODEL.WEIGHT_PATH = ""  # if resume from some checkpoint file
_C.MODEL.SAVE_CKPT = False

_C.MODEL.MODEL_ROOT = ""  # root folder for pretrained model weights

_C.MODEL.TYPE = "vit"
_C.MODEL.MLP_NUM = 0

_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.MLP_SIZES = []
_C.MODEL.LINEAR.DROPOUT = 0.1

# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT = CfgNode()
_C.MODEL.PROMPT.NUM_TOKENS = 5
_C.MODEL.PROMPT.LOCATION = "prepend"
# prompt initalizatioin: 
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_C.MODEL.PROMPT.INITIATION = "random"  # "final-cls", "cls-first12"
_C.MODEL.PROMPT.CLSEMB_FOLDER = ""
_C.MODEL.PROMPT.CLSEMB_PATH = ""
_C.MODEL.PROMPT.PROJECT = -1  # "projection mlp hidden dim"
_C.MODEL.PROMPT.DEEP = False # "whether do deep prompt or not, only for prepend location"


_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
_C.MODEL.PROMPT.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
_C.MODEL.PROMPT.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
# how to get the output emb for cls head:
    # original: follow the orignial backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"
_C.MODEL.PROMPT.DROPOUT = 0.0
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False
# VFPT / PromptedTransformer_Fourier (used when TRANSFER_TYPE == prompt_fourier)
_C.MODEL.PROMPT.FOURIER_TYPE = "fft"
_C.MODEL.PROMPT.FOURIER_DIMENSION = "all"
_C.MODEL.PROMPT.FOURIER_PERCENTAGE = 0.5
_C.MODEL.PROMPT.FOURIER_NUM = 100
_C.MODEL.PROMPT.FOURIER_FIRST_LAYER = True
_C.MODEL.PROMPT.FOURIER_ADDITION = False
_C.MODEL.PROMPT.FOURIER_LOCATION = "prepend"
_C.MODEL.PROMPT.FOURIER_ADDITION_NUM = 0
_C.MODEL.PROMPT.MIXED = False
_C.MODEL.PROMPT.FOURIER_HALF = "all"
# BPT / PromptedTransformer_BPT (when TRANSFER_TYPE == prompt_bilinear)
_C.MODEL.PROMPT.BPT_CHANNELS = 75
# "kaiming" (default, matches original bpt/Models): fan_out kaiming on conv, bias=False
# "normal": N(0,BPT_CONV_INIT_STD) on weight + zero bias; use for A/B ablations
_C.MODEL.PROMPT.BPT_CONV_INIT_MODE = "kaiming"
# When BPT_CONV_INIT_MODE == "normal"
_C.MODEL.PROMPT.BPT_CONV_INIT_STD = 1e-3
_C.MODEL.PROMPT.POSITION_EMBEDDINGS = True
_C.MODEL.PROMPT.VIS = False
_C.MODEL.PROMPT.VIS_JSON = ""
_C.MODEL.PROMPT.VIS_JSON_PROMPT = ""
_C.MODEL.PROMPT.VIS_JSON_FOURIER = "tta_vis"
# E2VPT (when TRANSFER_TYPE == prompt_vk)
_C.MODEL.P_VK = CfgNode()
_C.MODEL.P_VK.NUM_TOKENS_P = 5
_C.MODEL.P_VK.LOCATION = "prepend"
_C.MODEL.P_VK.INITIATION = "random"
_C.MODEL.P_VK.CLSEMB_FOLDER = ""
_C.MODEL.P_VK.CLSEMB_PATH = ""
_C.MODEL.P_VK.PROJECT = -1
_C.MODEL.P_VK.DEEP_P = True
_C.MODEL.P_VK.NUM_DEEP_LAYERS = None
_C.MODEL.P_VK.REVERSE_DEEP = False
_C.MODEL.P_VK.DEEP_SHARED = False
_C.MODEL.P_VK.FORWARD_DEEP_NOEXPAND = False
_C.MODEL.P_VK.VIT_POOL_TYPE = "original"
_C.MODEL.P_VK.DROPOUT_P = 0.0
_C.MODEL.P_VK.SAVE_FOR_EACH_EPOCH = False
_C.MODEL.P_VK.NUM_TOKENS = 5
_C.MODEL.P_VK.DEEP = True
_C.MODEL.P_VK.DROPOUT = 0.0
_C.MODEL.P_VK.LAYER_BEHIND = True
_C.MODEL.P_VK.SHARE_PARAM_KV = True
_C.MODEL.P_VK.ORIGIN_INIT = 2
_C.MODEL.P_VK.SHARED_ACCROSS = False
_C.MODEL.P_VK.MASK_CLS_TOKEN = False
_C.MODEL.P_VK.NORMALIZE_SCORES_BY_TOKEN = False
_C.MODEL.P_VK.CLS_TOKEN_MASK = False
_C.MODEL.P_VK.CLS_TOKEN_MASK_PERCENT_NUM = None
_C.MODEL.P_VK.CLS_TOKEN_MASK_PERCENT = [10, 20, 30, 40, 50, 60, 70, 80, 90]
_C.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN = 1
_C.MODEL.P_VK.CLS_TOKEN_MASK_PIECES = False
_C.MODEL.P_VK.CLS_TOKEN_PIECE_MASK_PERCENT_NUM = None
_C.MODEL.P_VK.CLS_TOKEN_PIECE_MASK_PERCENT = [10, 20, 30, 40, 50, 60, 70, 80, 90]
_C.MODEL.P_VK.MIN_NUMBER_CLS_TOKEN_PIECE = 4
_C.MODEL.P_VK.CLS_TOKEN_P_PIECES_NUM = 16
_C.MODEL.P_VK.MASK_RESERVE = False
_C.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_NUM = -1
_C.MODEL.P_VK.REWIND_MASK_CLS_TOKEN_PIECE_NUM = -1
_C.MODEL.P_VK.REWIND_STATUS = False
_C.MODEL.P_VK.REWIND_OUTPUT_DIR = ""
_C.MODEL.P_VK.SAVE_REWIND_MODEL = False
_C.MODEL.P_VK.MASK_CLS_TOKEN_ON_VK = False
_C.MODEL.P_VK.QUERY_PROMPT_MODE = 0
_C.MODEL.P_VK.FT_PT_MIXED = False
# ----------------------------------------------------------------------
# adapter options
# ----------------------------------------------------------------------
_C.MODEL.ADAPTER = CfgNode()
_C.MODEL.ADAPTER.REDUCATION_FACTOR = 8
_C.MODEL.ADAPTER.STYLE = "Pfeiffer"

# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 300


_C.SOLVER.SCHEDULER = "cosine"

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_MULTIPLIER = 1.              # for prompt + bias

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000


_C.SOLVER.DBG_TRAINABLE = False # if True, will print the name of trainable params

# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NAME = ""
_C.DATA.DATAPATH = ""
_C.DATA.FEATURE = ""  # e.g. inat2021_supervised

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = -1
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 224  # or 384

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True

_C.DIST_BACKEND = "nccl"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
