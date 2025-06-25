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
        
        # TopDownGetRandomScaleRotation에만 쓰임.
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        
        # TopDownRandomFlip에만 쓰임.
        self.flip = cfg.DATASET.FLIP
        
        # RandomShiftBboxCenter에만 쓰임.
        self.shift_factor = cfg.DATASET.SHIFT_FACTOR
        self.shift_prob = cfg.DATASET.SHIFT_PROB
        
        # HalfBodyTransform에만 쓰임.
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        
        # TopDownAffine에만 쓰임.
        self.use_udp = cfg.TEST.USE_UDP
        
        # TopDownGenerateHeatmap (_msra_generate_heatmap)에만 쓰임.
        self.is_target_heatmap = cfg.DATASET.TARGET_HEATMAP
        self.heatmap_type = cfg.MODEL.EXTRA.HEATMAP_TYPE
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        
        # ToTensor & NormalizeTensor에만 쓰임.
        self.transform = transform
        
        # ----------------------------------------------------------------------
        
        # 각 dataset에서 정의해야 하는 것. -------------------------------------------
        self.num_joints = 0 # NotImplementedError
        self.flip_pairs = [] # NotImplementedError
        self.db = [] # NotImplementedError
        # ----------------------------------------------------------------------

        # coco, mpii, joints 아무 곳에서도 쓰이지 않음. -------------------------------
        # self.output_path = cfg.OUTPUT_DIR # 일단 내부에서는 안 쓰임.
        # self.lower_body_ids = (11, 12, 13, 14, 15, 16) # 일단 내부에서는 안 쓰임.
        
        # for uncertainty
        # self.uncertainty = cfg.LOSS.UNCERTAINTY # 일단 내부에서는 안 쓰임.
        # self.normalized_map = cfg.LOSS.NORMALIZED_MAP # 일단 내부에서는 안 쓰임.
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

    def get(self, filepath: str) -> bytes:
        """Reads the contents of a file in binary mode and returns the data as bytes.

        Args:
            filepath (str): The path to the file to be read.

        Returns:
            bytes: The content of the file as a bytes object
        """
        filepath = str(filepath)
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf

    def _read_image(self, path: str) -> np.ndarray:
        """Reads the image file and returns the image as ndarrays.

        Args:
            path (str): The path to the image file to be read.

        Raises:
            ValueError: If the image is none after reading the file.

        Returns:
            np: Image as numpy object. The shape of this object is (H, W, C). The order of channel is 'rgb'
        """
        filepath = str(path)
        with open(filepath, "rb") as f:
            value_buf = f.read()
        img_bytes = value_buf
        img = imfrombytes(img_bytes, "color", channel_order="rgb") # bytes -> ndarrays로 바꿔줌. # (H, W, C:rgb)
        if img is None:
            raise ValueError(f"Fail to read {path}")
        return img

    def TopDownGetRandomScaleRotation(self, results: dict):
        _sf = self.scale_factor
        _rf = self.rotation_factor
        results['scale'] = results['scale'] * np.clip(np.random.randn() * _sf + 1, 1 - _sf, 1 + _sf)
        results['rotation'] = (
            np.clip(np.random.randn() * _rf, -_rf * 2, _rf * 2) if random.random() <= 0.6 else 0
        )
        
        return results
    
    def TopDownRandomFlip(self, results: dict):
        if self.flip and random.random() <= 0.5:
                
            # 이미지 좌우 뒤집기.
            results['data_numpy'] = results['data_numpy'][:, ::-1, :]
            
            # joints, joints_vis 좌우 뒤집기.
            results['joints'], results['joints_vis'] = fliplr_joints(
                results['joints'], results['joints_vis'], results['data_numpy'].shape[1], self.flip_pairs
            )
            
            # center도 좌우 뒤집기.
            results['center'][0] = results['data_numpy'].shape[1] - results['center'][0] - 1
        
        return results
    
    def RandomShiftBboxCenter(self, results: dict):
        results['center'] = results['center'] + np.random.uniform(-1, 1, 2) * self.shift_factor * results['scale'] * self.pixel_std
        return results
    
    def TopDownHalfBodyTransform(self, results: dict):
        if (
            np.sum(results['joints_vis'][:, 0]) > self.num_joints_half_body
            and np.random.rand() < self.prob_half_body
        ):
            _c_half_body, _s_half_body = self.half_body_transform(results['joints'], results['joints_vis'])
            if _c_half_body is not None and _s_half_body is not None:
                results['center'] = _c_half_body
                results['scale'] = _s_half_body
        return results
    
    def TopDownAffine(self, results: dict, image_size):
        """Affine transform the image to make input."""
        _joints_transformed = np.zeros_like(results['joints'])
        if self.use_udp:
            # 이 matrix를 만들 때, image size를 고려해서 변환 행렬이 만들어짐.
            # 이 matrix는 원본 이미지에서 원하는 image size로 변환해주는 행렬이다.
            trans = get_warp_matrix(results['rotation'], results['center'] * 2.0, image_size - 1.0, results['scale'] * 200.0)
            _input_img = cv2.warpAffine(
                results['data_numpy'],
                trans,
                (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
            _joints_transformed[:, 0:2] = warp_affine_joints(results['joints'][:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(results['center'], results['scale'], results['rotation'], self.image_size)
            _input_img = cv2.warpAffine(
                results['data_numpy'],
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
            for i in range(self.num_joints):
                if results['joints_vis'][i, 0] > 0.0:
                    _joints_transformed[i, 0:2] = affine_transform(results['joints'][i, 0:2], trans)
        
        results['input_img'] = _input_img
        results['joints_transformed'] = _joints_transformed
        
        return results
    
    def TopDownAffineForMultiResolution(self, results: dict):
        """Affine transform the image to make input."""
        
        results['input_img'] = []
        results['joints_transformed'] = []
        
        _joints_transformed = np.zeros_like(results['joints'])
        
        for idx, _image_size in enumerate(self.image_size):
            if self.use_udp:
                # 이 matrix를 만들 때, image size를 고려해서 변환 행렬이 만들어짐.
                # 이 matrix는 원본 이미지에서 원하는 image size로 변환해주는 행렬이다.
                trans = get_warp_matrix(results['rotation'], results['center'] * 2.0, _image_size - 1.0, results['scale'] * 200.0)
                _input_img = cv2.warpAffine(
                    results['data_numpy'],
                    trans,
                    (int(_image_size[0]), int(_image_size[1])),
                    flags=cv2.INTER_LINEAR,
                )
                _joints_transformed[:, 0:2] = warp_affine_joints(results['joints'][:, 0:2].copy(), trans)

            else:
                trans = get_affine_transform(results['center'], results['scale'], results['rotation'], _image_size)
                _input_img = cv2.warpAffine(
                    results['data_numpy'],
                    trans,
                    (int(_image_size[0]), int(_image_size[1])),
                    flags=cv2.INTER_LINEAR,
                )
                for i in range(self.num_joints):
                    if results['joints_vis'][i, 0] > 0.0:
                        _joints_transformed[i, 0:2] = affine_transform(results['joints'][i, 0:2], trans)
            
            results['input_img'].append(_input_img)
            results['joints_transformed'].append(_joints_transformed.copy())
        
        return results
    
    def TopDownGenerateHeatmap(self, results: dict, image_size, heatmap_size):
        if self.is_target_heatmap:
            _heatmaps, _heatmap_weights = self._msra_generate_heatmap(
                results['joints_transformed'],
                results['joints_vis'],
                image_size,
                heatmap_size,
                self.heatmap_type)
            _heatmaps = torch.from_numpy(_heatmaps)
            _heatmap_weights = torch.from_numpy(_heatmap_weights)
            results['heatmaps'] = _heatmaps
            results['heatmap_weights'] = _heatmap_weights
        
        return results
    
    def TopDownGenerateHeatmapForMutliResolution(self, results: dict):
        if self.is_target_heatmap:
            results['heatmaps'] = []
            results['heatmap_weights'] = []
            
            for idx, (_image_size, _heatmap_size) in enumerate(zip(self.image_size, self.heatmap_size)):
                _heatmaps, _heatmap_weights = self._msra_generate_heatmap(
                    results['joints_transformed'][idx],
                    results['joints_vis'],
                    _image_size,
                    _heatmap_size,
                    self.heatmap_type)
                
                _heatmaps = torch.from_numpy(_heatmaps)
                _heatmap_weights = torch.from_numpy(_heatmap_weights)
                
                results['heatmaps'].append(_heatmaps)
                results['heatmap_weights'].append(_heatmap_weights)
        
        return results
    
    
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        # ---------- LoadImageFromFile (준비 하기) ----------------------------------------
        # meta에만 들어갈 정보들
        image_file = db_rec["image"]
        filename = db_rec["filename"] if "filename" in db_rec else ""
        bbox = db_rec["bbox"] if "bbox" in db_rec else ""
        bbox_id = db_rec["bbox_id"] if "bbox_id" in db_rec else ""

        data_numpy = self._read_image(image_file) # 이미지 numpy (H, W, C) C->'rgb'
        if data_numpy is None:
            logger.error("=> fail to read {}".format(image_file))
            raise ValueError("Fail to read {}".format(image_file))

        joints: np.ndarray = db_rec["joints_2d"] # (num_joints, 2)
        joints_vis: np.ndarray = db_rec["joints_2d_vis"] # (num_joints, 2)

        center: np.ndarray = db_rec["center"] # (num_joints, 2)
        scale: np.ndarray = db_rec["scale"] # (2,)
        rotation = 0
        score = db_rec["score"] if "score" in db_rec else 1

        results = {
            'image_file': image_file,
            'filename': filename,
            'bbox': bbox,
            'bbox_id': bbox_id,
            'data_numpy': data_numpy,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': rotation,
            'score': score,
        }
        if self.is_train:
            results = self.TopDownGetRandomScaleRotation(results) # scale, rotation
            results = self.TopDownRandomFlip(results) # data_numpy, joints, joints_vis, center
            results = self.RandomShiftBboxCenter(results) # center
            results = self.TopDownHalfBodyTransform(results) # center, scale
        
        
        # self.is_target_keypoints, self.is_target_heatmap는 무조건 true라고 가정한다.
        if self.multi_res:
            results['target_joints'] = []
            
            results = self.TopDownAffineForMultiResolution(results) # input_img(list), joints_transformed(list)
            results = self.TopDownGenerateHeatmapForMutliResolution(results) # heatmaps(list), heatmap_weights(list)
            
            for idx, (input, joints) in enumerate(zip(results['input_img'], results['joints_transformed'])):
                # ToTensor & NormalizeTensor
                if self.transform:
                    results['input_img'][idx] = self.transform(input)
                if self.is_target_keypoints:
                    results['target_joints'].append(torch.from_numpy(joints))
            
            results['target_joints_vis'] = torch.from_numpy(results['joints_vis'][:, 1])
            
        else:
            # TopDownAffine
            results = self.TopDownAffine(results)
            
            # ToTensor & NormalizeTensor
            if self.transform:
                results['input_img'] = self.transform(results['input_img'])
            
            # TopDownGenerateHeatmap
            results = self.TopDownGenerateHeatmap(results)
                
            if self.is_target_keypoints:
                results['target_joints'] = torch.from_numpy(results['joints'])
                results['target_joints_vis'] = torch.from_numpy(results['joints_vis'][:, 1])
            
        meta = {
            "image": results['image_file'],
            "filename": results['filename'],
            # "joints": joints_mr if self.multi_res else results['joints'],
            "joints": results['joints_transformed'] if self.multi_res else results['joints'],
            "joints_vis": results['joints_vis'],
            "center": results['center'],
            "scale": results['scale'],
            "rotation": results['rotation'],
            "score": results['score'],
            "bbox": results['bbox'],
            "bbox_id": results['bbox_id'],
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
    
    # 현재는 msra style. # udp 적용 x
    def _msra_generate_heatmap(
        self,
        joints: np.ndarray,
        joints_vis: np.ndarray,
        image_size: list,
        heatmap_size: list,
        heatmap_type: str = "gaussian") -> tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            joints (np.ndarray): The position of body joints. (num_joints, 2)
            joints_visible (np.ndarray): The visibility of body joints. (num_joints, 2), visible: 1, invisible: 0.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                heatmap (num_joints, heatmap_height, heatmap_width),
                heatmap_weight (num_joints, 1). visible: 1, invisible: 0.
        """
        heatmap_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        heatmap_weight[:, 0] = joints_vis[:, 0]

        assert heatmap_type == "gaussian", "Only support gaussian map now!"

        if heatmap_type == "gaussian":
            heatmap = np.zeros(
                (self.num_joints, heatmap_size[1], heatmap_size[0]),
                dtype=np.float32,
            )

            # tmp_size는 heatmap의 반지름
            # 3-sigma rule # sigma에 3배를 하면 분포의 99.7% (거의 대부분)을 차지함.
            tmp_size = self.sigma * 3
            
            for joint_id in range(self.num_joints):
                feat_stride = image_size / heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5) # 기존의 joint x 좌표를 heatmap size에 맞게 조정(축소)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5) # 기존의 joint y 좌표를 heatmap size에 맞게 조정(축소)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # bl이 맞는 듯.
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] # ur이 맞는 듯
                if (
                    ul[0] >= heatmap_size[0]
                    or ul[1] >= heatmap_size[1]
                    or br[0] < 0
                    or br[1] < 0
                ):
                    # If not, just return the image as is
                    heatmap_weight[joint_id] = 0
                    continue

                # Generate gaussian filter
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                # heatmap을 그릴 plane에 gaussian heatmap을 그려줌.
                v = heatmap_weight[joint_id]
                if v > 0.5:
                    heatmap[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
                        g_y[0] : g_y[1], g_x[0] : g_x[1]
                    ]

        return heatmap, heatmap_weight

    def half_body_transform(self,
                            joints: np.ndarray,
                            joints_visible: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get center&scale for half-body transform.

        Args:
            joints (np.ndarray): The position of body joints. (num_joints, 2)
            joints_visible (np.ndarray): The visibility of body joints. (num_joints, 2), visible: 1, invisible: 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Return center and scale.
                The shape of center is (num_joints, 2), x,y positions.
                The shape of scale is (2,), x,y scales.
        """
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_visible[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            selected_joints = upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]# 각 joint의 선택된 좌표들의 평균값

        left_bottom = np.min(selected_joints, axis=0)
        right_top = np.max(selected_joints, axis=0)
        
        w = right_top[0] - left_bottom[0]
        h = right_top[1] - left_bottom[1]

        aspect_ratio = self.image_size[0][0] / self.image_size[0][1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.5
        return center, scale
    
    def random_cutmix(self, image, heatmap, MASK_HOLES_NUM=2):
        N, _, W_i, H_i = image.shape
        _, _, W_hm, H_hm = heatmap.shape
        
        device = image.device
        
        center_x = torch.randint(0, W_i, (N,MASK_HOLES_NUM), device=device).int()
        center_y = torch.randint(0, H_i, (N,MASK_HOLES_NUM), device=device).int()
        size = torch.randint(10, 20, (N,MASK_HOLES_NUM,2), device=device).int()
                
        center_x_hm = torch.round(center_x / 4).int()
        center_y_hm = torch.round(center_y / 4).int()
        size_hm = torch.round(size / 4).int()
        
        x0 = torch.clamp_(center_x-size[...,0],0,W_i)
        y0 = torch.clamp_(center_y-size[...,1],0,H_i)

        x1 = torch.clamp_(center_x+size[...,0],0,W_i)
        y1 = torch.clamp_(center_y+size[...,1],0,H_i)
        
        x0_hm = torch.clamp_(center_x_hm-size_hm[...,0],0,W_hm)
        y0_hm = torch.clamp_(center_y_hm-size_hm[...,1],0,H_hm)

        x1_hm = torch.clamp_(center_x_hm+size_hm[...,0],0,W_hm)
        y1_hm = torch.clamp_(center_y_hm+size_hm[...,1],0,H_hm)
        
        rand_index = torch.randperm(N).cuda()
        image_rand = image[rand_index]
        heatmap_rand = heatmap[rand_index]

        for i in range(N):
            for j in range(MASK_HOLES_NUM):
                image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = image_rand[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]]
                heatmap[i, :, y0_hm[i,j]:y1_hm[i,j], x0_hm[i,j]:x1_hm[i,j]] = heatmap_rand[i, :, y0_hm[i,j]:y1_hm[i,j], x0_hm[i,j]:x1_hm[i,j]]
        
        return image, heatmap