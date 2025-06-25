# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils.transforms import get_affine_transform
from lib.utils.transforms import affine_transform
from lib.utils.transforms import fliplr_joints
from lib.utils.post_transforms import get_warp_matrix
from lib.utils.post_transforms import warp_affine_joints
from lib.utils.io import imfrombytes
from typing import Tuple, Dict, Any, Union, Optional
from abc import ABC, abstractmethod
from lib.dataset.pipelines import (
    TopDownGetRandomScaleRotation,
    TopDownRandomFlip,
    RandomShiftBboxCenter,
    TopDownHalfBodyTransform,
    TopDownAffine,
    TopDownGenerateHeatmap,
    LoadImageFromFile
    )
from lib.dataset.shared_transform import ToTensor, NormalizeTensor

logger = logging.getLogger(__name__)


# class JointsDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]):
class JointsDataset(Dataset):
    def __init__(
        self,
        cfg,
        root,
        image_set,
        is_train,
        dataset_idx=0,
        transform=None
    ):
        # ----------------------------------------------------------------------
        # joints_dataset에서 정의해야 하는 것 (general puropose)
        self.pixel_std = 200 # RandomShiftBboxCenter에서만 필요 / coco에서 scale 계산할 때도 필요함.
        self.is_train = is_train
        self.root = root # 각 데이터셋에서 이 root를 받아서 씀.
        self.image_set = image_set # 각 데이터셋에서 이 image_set을 받아서 씀.
        self.multi_res = False
        self.image_size: list[Union[list, int]] = cfg.MODEL.IMAGE_SIZE
        self.heatmap_size: list[Union[list, int]] = cfg.MODEL.HEATMAP_SIZE
        if isinstance(cfg.MODEL.IMAGE_SIZE[0], (np.ndarray, list)): # 이 방식은 cfg에 의존해서 안좋음.
            self.multi_res = True
        # for uncertainty
        self.is_target_keypoints = cfg.DATASET.TARGET_KEYPOINT
        
        # LoadImageFromFile
        self.load_image = LoadImageFromFile(logger)
        
        # TopDownGetRandomScaleRotation에만 쓰임.
        self.get_random_scale_rotation = TopDownGetRandomScaleRotation(
            scale_factor=cfg.DATASET.SCALE_FACTOR,
            rotation_factor=cfg.DATASET.ROT_FACTOR
        )
        
        # TopDownRandomFlip에만 쓰임.
        self.flip = cfg.DATASET.FLIP
        self.random_flip = TopDownRandomFlip()
        
        # RandomShiftBboxCenter에만 쓰임.
        self.random_shift_bbox_center = RandomShiftBboxCenter(
            shift_factor=cfg.DATASET.SHIFT_FACTOR,
            pixel_std=self.pixel_std
        )
        
        # HalfBodyTransform에만 쓰임.
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.half_body_transform = TopDownHalfBodyTransform(
            num_joints_half_body=cfg.DATASET.NUM_JOINTS_HALF_BODY,
            prob_half_body=cfg.DATASET.PROB_HALF_BODY
        )
        
        # TopDownAffine에만 쓰임.
        self.use_udp = cfg.TEST.USE_UDP
        self.affine = TopDownAffine(
            use_udp=self.use_udp
        )
        
        # TopDownGenerateHeatmap (_msra_generate_heatmap)에만 쓰임.
        self.is_target_heatmap = cfg.DATASET.TARGET_HEATMAP
        self.generate_heatmap = TopDownGenerateHeatmap(
            sigma=cfg.MODEL.EXTRA.SIGMA,
            heatmap_type=cfg.MODEL.EXTRA.HEATMAP_TYPE
        )
        
        # ToTensor & NormalizeTensor에만 쓰임.
        self.transform = transform
        self.to_tensor = ToTensor()
        self.normalize_tensor = NormalizeTensor()
        
        # ----------------------------------------------------------------------
        
        # 각 dataset에서 정의해야 하는 것. -------------------------------------------
        self.num_joints = 0 # NotImplementedError
        self.flip_pairs = [] # NotImplementedError
        self.db = [] # NotImplementedError
        # ----------------------------------------------------------------------
    
    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError
    
    # @abstractmethod
    # def evaluate(self):
    #     pass

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        # 아마 joints_dataset_new.py에서 results를 계속해서 넘길 때, 'joints' 부분이 잘못 들어가고 있었던 것 같다.
        results = copy.deepcopy(self.db[idx]) # results가 dict인가?
        results['ann_info'] = {
            'num_joints': self.num_joints,
            'upper_body_ids': self.upper_body_ids,
            'image_size': self.image_size,
            'heatmap_size': self.heatmap_size,
            'flip_pairs': self.flip_pairs
        }
        
        results = self.load_image(results)
        
        if self.is_train:
            results = self.get_random_scale_rotation(results) # scale, rotation
            if self.flip:
                results = self.random_flip(results) # data_numpy, joints, joints_vis, center
            results = self.random_shift_bbox_center(results) # center
            results = self.half_body_transform(results) # center, scale
        
        results = self.affine(results)
        if self.is_target_heatmap:
            results = self.generate_heatmap(results) # heatmaps(list), heatmap_weights(list)
        results = self.to_tensor(results)
        results = self.normalize_tensor(results)
        
        if self.is_target_keypoints:
            if self.multi_res:
                results['target_joints'] = [joints for joints in results['joints_transformed']]
                results['target_joints_vis'] = torch.from_numpy(results['joints_2d_vis'][:, 1])
            else:
                results['target_joints'] = torch.from_numpy(results['joints_2d'])
                results['target_joints_vis'] = torch.from_numpy(results['joints_2d_vis'][:, 1])
            
        meta = {
            "image": results['image'],
            "filename": results['filename'] if 'filename' in results else "",
            "joints": results['joints_transformed'],
            "joints_vis": results['joints_2d_vis'],
            "center": results['center'],
            "scale": results['scale'],
            "rotation": results['rotation'],
            "score": results['score'] if 'score' in results else 1,
            "bbox": results['bbox'] if 'bbox' in results else 1,
            "bbox_id": results['bbox_id'] if 'bbox_id' in results else "",
            "flip_pairs": self.flip_pairs,
        }

        return (
            results['input_img'],
            results['target_joints'],
            results['target_joints_vis'],
            results['heatmaps'],
            results['heatmap_weights'],
            meta,
        )