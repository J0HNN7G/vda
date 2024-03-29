# -*- coding: utf-8 -*-
# This file is code altered from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/same_vis_same_phys-spynet-resnet18-pybullet"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/same_vis_same_phys"
_C.DATASET.list_train = "./data/same_vis_same_phys/train"
_C.DATASET.list_val = "./data/same_vis_same_phys/val"
_C.DATASET.img_size = (256, 256)
_C.DATASET.buffer_size = 3
_C.DATASET.num_classes = 6
# simulation frames per second
_C.DATASET.fps = 30
# where all URDF files are stored
_C.DATASET.urdf_folder = './data'

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of optical flow
_C.MODEL.optical_flow = "spynet"
# architecture of perceptual model
_C.MODEL.perceptual = "resnet18"
# weights to finetune perceptual model
_C.MODEL.weights_perceptual = ""
# include images in training
_C.MODEL.include_images = True
# include optical flow
_C.MODEL.include_optical_flow = True

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_perceptual = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304

# early stopping if no improvements in this many iterations
_C.TRAIN.no_improv_limit = 10


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_5.pth"
# prediction steps to
_C.VAL.prediction_timesteps = [1, 5, 10, 20, 40, 50, 57]

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_5.pth"
# folder to output visualization results
_C.TEST.result = "./results"
# physics engine for dynamics
_C.TEST.physics_engine = "pybullet"
# graphics engine for rendering
_C.TEST.graphics_engine = "pybullet"
# prediction steps to output files for
_C.TEST.prediction_timesteps = [1, 5, 10, 20]
