# Modified by Sukmin Yun (sukmin.yun@kaist.ac.kr)
#
"""Configs."""
from fvcore.common.config import CfgNode
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.ERM_TRAIN = CfgNode()
_C.ERM_TRAIN.ENABLE = False
_C.ERM_TRAIN.DATASET = ''
_C.ERM_TRAIN.BATCH_SIZE = 8
_C.ERM_TRAIN.EVAL_PERIOD = 1
_C.ERM_TRAIN.CHECKPOINT_PERIOD = 1
_C.ERM_TRAIN.AUTO_RESUME = True
_C.ERM_TRAIN.FP16 = False
_C.ERM_TRAIN.CHECKPOINT_FILE_PATH = ""
_C.ERM_TRAIN.FINETUNE = False
# If True, reset epochs when loading checkpoint.
_C.ERM_TRAIN.CHECKPOINT_EPOCH_RESET = False
# If set, clear all layer names according to the pattern provided.
_C.ERM_TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)
_C.ERM_TRAIN.NUM_ITERS = 1


_C.ERM_TEST = CfgNode()
_C.ERM_TEST.ENABLE = False
_C.ERM_TEST.DATASET = ''
_C.ERM_TEST.BATCH_SIZE = 8
_C.ERM_TEST.CHECKPOINT_FILE_PATH = ""
_C.ERM_TEST.NUM_SPATIAL_CROPS = 3
_C.ERM_TEST.NUM_ENSEMBLE_VIEWS = 1
_C.ERM_TEST.SAVE_RESULTS_PATH = ""


# ----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name
_C.MODEL.MODEL_NAME = ""
_C.MODEL.FC_ZERO_INIT = True

# Loss function.
_C.MODEL.LOSS_FUNC = "mse"
_C.MODEL.NUM_CLASSES = 0

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.PATH_TO_ANNOTATION = ""
_C.DATA.PATH_TO_RAWDATA = ""
_C.DATA.PATH_TO_JPEG = ""
_C.DATA.PATH_TO_SIREN = ""
_C.DATA.PATH_TO_SIREN_DERIVATIVE = ""
_C.DATA.PATH_TO_PREPROCESSED = ""
_C.DATA.PATH_TO_PREPROCESSED_TMP = ""

_C.DATA.TRAIN_SPLIT_DIR = ""
_C.DATA.VAL_SPLIT_DIR = ""
_C.DATA.TEST_SPLIT_DIR = ""

_C.DATA.INDEX_FROM = 0
_C.DATA.INDEX_TO = None

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224
# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 224

# use random augment
_C.DATA.USE_RAND_AUGMENT = False

_C.DATA.ENSEMBLE_METHOD = 'sum'

_C.DATA.NUM_FRAMES = 8


_C.DATA.APPEND_TO_OUTPUT_DIRNAME = []

_C.DATA.FRAMES_SELECTION = 'all' # 'initial' | 'center' | 'last' | 'random' | 'all' < should be used with DATA.PATCH_SELECTION
_C.DATA.PATCH_SELECTION = 'max_gradient' # 'max_gradient' | 'max_difference' | 'random'
_C.DATA.OUTPUT_FEATURE = 'rgb' # 'rgb' | 'rgb+gradient | 'rgb+difference'

# The std value of the video raw pixels across the R G B channels.
# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
_C.DATA.STD = [0.225, 0.225, 0.225]
_C.DATA.TIME_RANGE = 0.3

_C.DATA.INV_UNIFORM_SAMPLE = True
_C.DATA.REVERSE_INPUT_CHANNEL = False

# motionformer options

_C.VIT = CfgNode()

_C.VIT.PATCH_SIZE = 16  # Patch-size spatial to tokenize input
_C.VIT.PATCH_SIZE_TEMP = 2  # Patch-size temporal to tokenize input
_C.VIT.CHANNELS = 3  # Number of input channels
_C.VIT.EMBED_DIM = 768  # Embedding dimension
_C.VIT.DEPTH = 12  # Depth of transformer: number of layers
_C.VIT.NUM_HEADS = 12  # number of attention heads
_C.VIT.MLP_RATIO = 4  # expansion ratio for MLP
_C.VIT.QKV_BIAS = True  #add bias to QKV projection layer
_C.VIT.VIDEO_INPUT = True  # video input
_C.VIT.TEMPORAL_RESOLUTION = 8  # temporal resolution i.e. number of frames
_C.VIT.USE_MLP = False  # use MLP classification head
_C.VIT.DROP = 0.0  # Dropout rate for
_C.VIT.DROP_PATH = 0.0  #Stochastic drop rate
_C.VIT.HEAD_DROPOUT = 0.0  # Dropout rate for MLP head
_C.VIT.POS_DROPOUT = 0.0  #Dropout rate for positional embeddings
_C.VIT.ATTN_DROPOUT = 0.0  #Dropout rate
_C.VIT.HEAD_ACT = "tanh"  #Activation for head
_C.VIT.IM_PRETRAINED = True  #Use IM pretrained weights
_C.VIT.PRETRAINED_WEIGHTS = "vit_1k"  #Pretrained weights type
_C.VIT.POS_EMBED = "separate"  # Type of position embedding
_C.VIT.ATTN_LAYER = "trajectory"  # Self-Attention layer
_C.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE = True  # Flag to use original trajectory attn code
_C.VIT.APPROX_ATTN_TYPE = "none"  # Approximation type
_C.VIT.APPROX_ATTN_DIM = 128  # Approximation Dimension


# -----------------------------------------------------------------------------
# XVIT options
# -----------------------------------------------------------------------------
_C.XVIT = CfgNode()
_C.XVIT.NUM_SEGMENTS = 8
_C.XVIT.CONSENSUS_TYPE = "vit"
_C.XVIT.BEFORE_SOFTMAX = True
_C.XVIT.USE_XVIT = False
_C.XVIT.SHIFT_DIV = 8
_C.XVIT.BASE_MODEL = "resnet50"
_C.XVIT.PRETRAIN = True
_C.XVIT.LOCATIONS_LIST = [9, 10, 11]
_C.XVIT.BACKBONE = CfgNode()
_C.XVIT.BACKBONE.DROP_PATH_RATE = 0.0
_C.XVIT.BACKBONE.DROP_ATTN_RATE = 0.0
_C.XVIT.BACKBONE.NORM_LAYER = "LN"  # GN, IN
_C.XVIT.DROPOUT_RATE = 0.5

# -----------------------------------------------------------------------------
# VTN options (also XVIT option)
# -----------------------------------------------------------------------------
_C.TEMPORAL_HEAD = CfgNode()
_C.TEMPORAL_HEAD.HIDDEN_DIM = 768
# Longformer: number of attention heads for each attention layer in the Transformer encoder.
_C.TEMPORAL_HEAD.NUM_ATTENTION_HEADS = 12
# Longformer: number of hidden layers in the Transformer encoder.
_C.TEMPORAL_HEAD.NUM_HIDDEN_LAYERS = 1
# Longformer: The dropout ratio for the attention probabilities.
_C.TEMPORAL_HEAD.ATTENTION_PROBS_DROPOUT_PROB = 0.0
# Longformer: The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
_C.TEMPORAL_HEAD.HIDDEN_DROPOUT_PROB = 0.0
# MLP Head: the dimension of the MLP head hidden layer.
_C.TEMPORAL_HEAD.MLP_DIM = 768

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()
# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False
# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200
# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0
# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1
# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 5e-3

_C.SOLVER.SMOOTHING = 0.0

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 10

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# Adam AMSGRAD.
_C.SOLVER.AMSGRAD = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

_C.SOLVER.USE_MIXED_PRECISION = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

_C.MP_SPAWN = True
_C.PRINT_ALL_PROCS = False

_C.GPU_ID = None

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 4

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./experiments/"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 0

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = False

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 100

# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 1

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.ERM_TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""
# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
# This file must be provided to enable plotting confusion matrix
# by a subset or parent categories.
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# Path to a json file for categories -> classes mapping
# in the format {"parent_class": ["child_class1", "child_class2",...], ...}.
_C.TENSORBOARD.CATEGORIES_PATH = ""

# Config for confusion matrices visualization.
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# Visualize confusion matrix.
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# Figure size of the confusion matrices plotted.
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# Path to a subset of categories to visualize.
# File contains class names separated by newline characters.
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# Config for histogram visualization.
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# Visualize histograms.
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# Path to a subset of classes to plot histograms.
# Class names must be separated by newline characters.
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# Visualize top-k most predicted classes on histograms for each
# chosen true label.
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# Figure size of the histograms plotted.
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]

# Config for layers' weights and activations visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS = CfgNode()

# If False, skip model visualization.
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

# If False, skip visualizing model weights.
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

# If False, skip visualizing model activations.
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

# If False, skip visualizing input videos.
_C.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False


# List of strings containing data about layer names and their indexing to
# visualize weights and activations for. The indexing is meant for
# choosing a subset of activations outputed by a layer for visualization.
# If indexing is not specified, visualize all activations outputed by the layer.
# For each string, layer name and indexing is separated by whitespaces.
# e.g.: [layer1 1,2;1,2, layer2, layer3 150,151;3,4]; this means for each array `arr`
# along the batch dimension in `layer1`, we take arr[[1, 2], [1, 2]]
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
# Top-k predictions to plot on videos
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
# Config for visualization video inputs with Grad-CAM.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
# Whether to run visualization using Grad-CAM technique.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
# CNN layers to use for Grad-CAM. The number of layers must be equal to
# number of pathway(s).
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
# If True, visualize Grad-CAM using true labels for each instances.
# If False, use the highest predicted class.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

# Config for visualization for wrong prediction visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
# Folder tag to origanize model eval videos under.
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified videos."
# Subset of labels to visualize. Only wrong predictions with true labels
# within this subset is visualized.
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""

def _assert_and_infer_cfg(cfg):
    # TRAIN assertions.
    assert cfg.ERM_TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.ERM_TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    return cfg

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
