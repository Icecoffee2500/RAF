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
import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Union
from abc import ABC, abstractmethod
from dataset.pipelines_future import (
    TopDownGetRandomScaleRotation,
    TopDownRandomFlip,
    RandomShiftBboxCenter,
    TopDownHalfBodyTransform,
    TopDownAffine,
    TopDownGenerateHeatmap,
    LoadImageFromFile
    )
from dataset.shared_transform import ToTensor, NormalizeTensor

logger = logging.getLogger(__name__)


# class JointsDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]):
class  JointsDataset(Dataset):
    def __init__(
        self,
        cfg,
        root,
        image_set,
        image_size: list[Union[list, int]],
        heatmap_size: list[Union[list, int]],
        is_train,
        transform=None
    ):
        # ----------------------------------------------------------------------
        # joints_dataset에서 정의해야 하는 것 (general puropose)
        self.pixel_std = 200 # RandomShiftBboxCenter에서만 필요 / coco에서 scale 계산할 때도 필요함.
        self.is_train = is_train
        self.root = root # 각 데이터셋에서 이 root를 받아서 씀.
        self.image_set = image_set # 각 데이터셋에서 이 image_set을 받아서 씀.
        self.multi_res = False
        # self.image_size: list[Union[list, int]] = cfg.MODEL.IMAGE_SIZE
        # self.heatmap_size: list[Union[list, int]] = cfg.MODEL.HEATMAP_SIZE
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        
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
            pixel_std=self.pixel_std,
            shift_prob=cfg.DATASET.SHIFT_PROB
        )
        
        # HalfBodyTransform에만 쓰임.
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.half_body_transform = TopDownHalfBodyTransform(
            upper_body_ids=self.upper_body_ids,
            image_size=self.image_size,
            num_joints_half_body=cfg.DATASET.NUM_JOINTS_HALF_BODY,
            prob_half_body=cfg.DATASET.PROB_HALF_BODY
        )
        
        # TopDownAffine에만 쓰임.
        self.affine = TopDownAffine(
            image_size=self.image_size,
            use_udp=cfg.TEST.USE_UDP
        )
        
        # TopDownGenerateHeatmap (_msra_generate_heatmap)에만 쓰임.
        self.is_target_heatmap = cfg.DATASET.TARGET_HEATMAP
        self.generate_heatmap = TopDownGenerateHeatmap(
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=cfg.MODEL.EXTRA.SIGMA,
            heatmap_type=cfg.MODEL.EXTRA.HEATMAP_TYPE
        )
        
        # ToTensor & NormalizeTensor에만 쓰임.
        self.transform = transform
        self.to_tensor = ToTensor()
        self.normalize_tensor = NormalizeTensor()
        
        # 각 dataset에서 정의해야 하는 것. -------------------------------------------
        self.num_joints = 0 # NotImplementedError
        self.flip_pairs = [] # NotImplementedError
        self.db = [] # NotImplementedError
    
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
        data = copy.deepcopy(self.db[idx])
        
        # initial variables
        image_file = data.get('image', None)
        filename = data['filename'] if 'filename' in data else ''
        bbox = data['bbox'] if 'bbox' in data else ''
        bbox_id = data['bbox_id'] if 'bbox_id' in data else ''
        joints: np.ndarray = data['joints_2d'] # (num_joints, 2)
        joints_vis: np.ndarray = data['joints_2d_vis'] # (num_joints, 2)
        center: np.ndarray = data['center'] # (num_joints, 2)
        scale: np.ndarray = data['scale'] # (2,)
        score = data['score'] if 'score' in data else 1
        rotation = 0
        
        img = self.load_image(image_file)
        if self.is_train:
            scale, rotation = self.get_random_scale_rotation(scale)
            
            if self.flip:
                img, joints, joints_vis, center = self.random_flip(img, joints, joints_vis, center, self.flip_pairs)
            
            center = self.random_shift_bbox_center(center, scale)
            
            c_half_body, s_half_body = self.half_body_transform(self.num_joints, joints, joints_vis)
            if c_half_body is not None and s_half_body is not None:
                center = c_half_body
                scale = s_half_body
        
        input_img, joints_transformed = self.affine(img, joints, joints_vis, center, scale, rotation, self.num_joints)
        input_img = self.to_tensor(input_img)
        input_img = self.normalize_tensor(input_img)
        
        if self.is_target_heatmap:
            heatmaps, heatmap_weights = self.generate_heatmap(joints_transformed, joints_vis, self.num_joints)
        
        if self.is_target_keypoints:
            if self.multi_res:
                target_joints = [joints for joints in joints_transformed]
                target_joints_vis = torch.from_numpy(joints_vis[:, 1])
            else:
                target_joints = torch.from_numpy(joints_transformed)
                target_joints_vis = torch.from_numpy(joints_vis[:, 1])
            
        meta = {
            "image": image_file,
            "filename": filename,
            "joints": joints_transformed,
            "joints_vis": joints_vis,
            "center": center,
            "scale": scale,
            "rotation": rotation,
            "score": score,
            "bbox": bbox,
            "bbox_id": bbox_id,
            "flip_pairs": self.flip_pairs,
        }

        return (
            input_img,
            target_joints,
            target_joints_vis,
            heatmaps,
            heatmap_weights,
            meta,
        )