# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# 23. 05. 15 수정사항
# 1. simple baseline
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.OUTPUT_DIR = ""
config.LOG_DIR = ""
config.GPUS = "0"
config.WORKERS = 4
config.PRINT_FREQ = 20

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = "pose_resnet"
config.MODEL.TYPE = ""
config.MODEL.PRETRAINED = ""
config.MODEL.NUM_JOINTS = 16
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.HEATMAP_SIZE = [64, 64]

#------------------------------------------------#
config.MODEL.EXTRA = edict()
config.MODEL.EXTRA.SIGMA = 2
config.MODEL.EXTRA.HEATMAP_TYPE = "gaussian"
config.MODEL.FREEZE_NAME = ""
config.MODEL.DIFF_NAME = ""
config.MODEL.USE_EXP_KP = False
config.MODEL.USE_AMP = False

config.MODEL.USE_GPE = False
config.MODEL.USE_LPE = False
config.MODEL.USE_GAP = False

config.LOSS = edict()
# config.LOSS.NAME = ""
config.LOSS.HM_LOSS = ""
config.LOSS.KD_LOSS = ""
config.LOSS.USE_TARGET_WEIGHT = True
# Uncertainty 관련 설정 제거됨 (사용하지 않음)
# config.LOSS.USE_CROSS_HM = False
config.LOSS.HM_LOSS_WEIGHT = 0
config.LOSS.KP_LOSS_WEIGHT = 0
config.LOSS.KD_LOSS_WEIGHT = 0

# DATASET related params
config.DATASET = edict()

config.DATASET_SETS = None

config.DATASET.ROOT = ""
config.DATASET.DATASET = "mpii"
config.DATASET.TRAIN_SET = "train"
config.DATASET.TEST_SET = "valid"
config.DATASET.TARGET_HEATMAP = False

# training data augmentation
config.DATASET.FLIP = True
config.DATASET.SCALE_FACTOR = 0.5
config.DATASET.ROT_FACTOR = 40
#------------------------------------------------#
config.DATASET.SHIFT_FACTOR = 0.12
config.DATASET.SHIFT_PROB = 0.3
config.DATASET.SELECT_DATA = False
config.DATASET.NUM_JOINTS_HALF_BODY = 8
config.DATASET.PROB_HALF_BODY = 0.3
#------------------------------------------------#
config.DATASET.NUMBER_OF_SPLITS = 1
#------------------------------------------------#

# Fed related params
config.FED = edict()
config.FED.FEDAVG = True
config.FED.FEDPROX = False
config.FED.MU = 0.0

# KD related params
config.KD_USE = False
config.KD_ALPHA = 1.0
config.LOSS_SCALE = 1

#------------------------------------------------#
# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001
config.TRAIN.OPTIMIZER = "adam"
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 211
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True
# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
config.TEST.FLIP_TEST = False

config.TEST.USE_GT_BBOX = False
#------------------------------------------------#
config.TEST.USE_UDP = False
#------------------------------------------------#
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ""
config.TEST.BBOX_THRE = 1.0
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = False
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False
#------------------------------------------------#
config.EVALUATION = edict()
config.EVALUATION.INTERVAL = 10
#------------------------------------------------#

# 핵심함수: YAML 파일을 읽어서 config 객체에 업데이트함.
def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == "SCALES":
                        config[k][0] = tuple(v)
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

# update_config 함수에서 사용되는 내부 함수.
def _update_dict(k, v):
    if k == "DATASET":
        if "MEAN" in v and v["MEAN"]:
            v["MEAN"] = np.array([eval(x) if isinstance(x, str) else x for x in v["MEAN"]])
        if "STD" in v and v["STD"]:
            v["STD"] = np.array([eval(x) if isinstance(x, str) else x for x in v["STD"]])
    if k == "MODEL":
        if "EXTRA" in v and "HEATMAP_SIZE" in v["EXTRA"]:
            if isinstance(v["EXTRA"]["HEATMAP_SIZE"], int):
                v["EXTRA"]["HEATMAP_SIZE"] = np.array(
                    [v["EXTRA"]["HEATMAP_SIZE"], v["EXTRA"]["HEATMAP_SIZE"]]
                )
            else:
                v["EXTRA"]["HEATMAP_SIZE"] = np.array(v["EXTRA"]["HEATMAP_SIZE"])
        if "IMAGE_SIZE" in v:
            if isinstance(v["IMAGE_SIZE"], int):
                v["IMAGE_SIZE"] = np.array([v["IMAGE_SIZE"], v["IMAGE_SIZE"]])
            else:
                v["IMAGE_SIZE"] = np.array(v["IMAGE_SIZE"])
        if "PR_IMAGE_SIZE" in v:
            if isinstance(v["PR_IMAGE_SIZE"], int):
                v["PR_IMAGE_SIZE"] = np.array([v["PR_IMAGE_SIZE"], v["PR_IMAGE_SIZE"]])
            else:
                v["PR_IMAGE_SIZE"] = np.array(v["PR_IMAGE_SIZE"])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


# Checkpoint 저장할 때 사용되는 모델 이름을 정하는 함수.
#TODO: 모델 이름 수정해야 함.
def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    if "vit" in name:
        if isinstance(cfg.MODEL.IMAGE_SIZE[0], (list, np.ndarray)):
            name = f"{name}_{cfg.MODEL.TYPE}"
            full_name = ""
            for i in range(len(cfg.MODEL.IMAGE_SIZE)):
                full_name += f"{cfg.MODEL.IMAGE_SIZE[i][1]}x{cfg.MODEL.IMAGE_SIZE[i][0]}_"
            full_name += name
        else:
            name = "{model}_{type}".format(model=name, type=cfg.MODEL.TYPE)
            full_name = "{height}x{width}_{name}".format(
                height=cfg.MODEL.IMAGE_SIZE[1], width=cfg.MODEL.IMAGE_SIZE[0], name=name
            )
    else:
        raise ValueError("Unkown model: {}".format(cfg.MODEL))

    return name, full_name