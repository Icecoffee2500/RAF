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

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.post_transforms import get_warp_matrix
from utils.post_transforms import warp_affine_joints
from utils.io import imfrombytes
from typing import Tuple, Dict, Any, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# class JointsDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]):
class JointsDataset(Dataset):
    # def __init__(self, cfg, root, image_set, is_train, dataset_idx=0, transform=None):
    def __init__(
        self,
        cfg,
        root,
        image_set,
        is_train,
        dataset_idx=0,
        transform=None
    ):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set
        self.db = []
        
        self.output_path = cfg.OUTPUT_DIR # 일단 내부에서는 안 쓰임.
        
        # if 'proxy' in self.root:
        #     self.dataset_sets = cfg.DATASET_PROXY
        # elif cfg.DATASET_SETS is not None:
        #     self.dataset_sets = cfg.DATASET_SETS[dataset_idx]
        # else:
        #     self.dataset_sets = cfg.DATASET

        # self.dataset_name = cfg.DATASET.DATASET
        # self.dataset_name = self.dataset_sets.DATASET
        
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
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        # image size & heatmap size
        if isinstance(cfg.MODEL.IMAGE_SIZE[0], (np.ndarray, list)):
            self.image_size = cfg.MODEL.IMAGE_SIZE
        else:
            self.image_size = [cfg.MODEL.IMAGE_SIZE]
        # self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        if isinstance(cfg.MODEL.HEATMAP_SIZE[0], (np.ndarray, list)):
            self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        else:
            self.heatmap_size = [cfg.MODEL.HEATMAP_SIZE]
        
        # TopDownAffine에만 쓰임.
        self.use_udp = cfg.TEST.USE_UDP

        # TopDownGenerateTarget (_msra_generate_heatmap)에만 쓰임.
        self.heatmap_type = cfg.MODEL.EXTRA.HEATMAP_TYPE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        # ToTensor & NormalizeTensor에만 쓰임.
        self.transform = transform
        
        # for uncertainty
        self.uncertainty = cfg.LOSS.UNCERTAINTY # 일단 내부에서는 안 쓰임.
        self.normalized_map = cfg.LOSS.NORMALIZED_MAP # 일단 내부에서는 안 쓰임.
        self.is_target_keypoints = cfg.DATASET.TARGET_KEYPOINT
        
        # target을 heatmap으로 할지 결정
        self.is_target_heatmap = cfg.DATASET.TARGET_HEATMAP
        

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

    def __getitem__(self, idx):
        # ---------- 준비 하기 ----------------------------------------
        db_rec = copy.deepcopy(self.db[idx])

        # meta에만 들어갈 정보들
        image_file = db_rec["image"]
        filename = db_rec["filename"] if "filename" in db_rec else ""
        bbox = db_rec["bbox"] if "bbox" in db_rec else ""
        bbox_id = db_rec["bbox_id"] if "bbox_id" in db_rec else ""

        # sigma = db_rec["sigma"] if "sigma" in db_rec else ""
        idx_ = db_rec["idx"] if "idx" in db_rec else ""

        data_numpy = self._read_image(image_file) # 이미지 numpy (H, W, C) C->'rgb'
        if data_numpy is None:
            logger.error("=> fail to read {}".format(image_file))
            raise ValueError("Fail to read {}".format(image_file))

        joints: np.ndarray = db_rec["joints_2d"] # (num_joints, 2)
        joints_vis: np.ndarray = db_rec["joints_2d_vis"] # (num_joints, 2)

        center: np.ndarray = db_rec["center"] # (num_joints, 2)
        scale: np.ndarray = db_rec["scale"] # (2,)
        score = db_rec["score"] if "score" in db_rec else 1
        rotation = 0
        
        # ---------- pipeline ----------------------------------------

        if self.is_train:
            # TopDownGetRandomScaleRotation ----------------------------------------
            # scale_factor, rotation factor를 받아서 scale과 rotation을 반환하는 함수
            # scale, rotation = TopDownGetRandomScaleRotation(self.scale_factor, self.rotation_factor)
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = (
                np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
            )
            # TopDownRandomFlip ----------------------------------------
            # 랜덤하게 이미지와 joints, joints_vis의 좌우를 뒤집음.
            # data_numpy, joints, joints_vis, center = TopDownRandomFlip(data_numpy, joints, joints_vis, center, flip=self.flip)
            if self.flip and random.random() <= 0.5:
                
                # 이미지 좌우 뒤집기.
                data_numpy = data_numpy[:, ::-1, :]
                
                # joints, joints_vis 좌우 뒤집기.
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs
                )
                
                # center도 좌우 뒤집기.
                center[0] = data_numpy.shape[1] - center[0] - 1

            # RandomShiftBboxCenter ----------------------------------------
            # pixel_std: float = 200.0
            if np.random.rand() < self.shift_prob:
                center += np.random.uniform(-1, 1, 2) * self.shift_factor * scale * self.pixel_std

            # Half body transform ----------------------------------------
            if (
                np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body
            ):
                c, s = self.half_body_transform(joints, joints_vis)
                if c is not None and s is not None:
                    center = c
                    scale = s

        # ------------------------------------------------------------------------------------------
        # # change #
        # # TopDownAffine
        # if self.use_udp:
        #     _trans = get_warp_matrix(rotation, center * 2.0, self.image_size - 1.0, scale * 200.0)
        #     input_img = cv2.warpAffine(
        #         data_numpy,
        #         _trans,
        #         (int(self.image_size[0]), int(self.image_size[1])),
        #         flags=cv2.INTER_LINEAR,
        #     )
        #     joints[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), _trans)

        # else:
        #     _trans = get_affine_transform(center, scale, rotation, self.image_size)
        #     input_img = cv2.warpAffine(
        #         data_numpy,
        #         _trans,
        #         (int(self.image_size[0]), int(self.image_size[1])),
        #         flags=cv2.INTER_LINEAR,
        #     )
        #     for i in range(self.num_joints):
        #         if joints_vis[i, 0] > 0.0:
        #             joints[i, 0:2] = affine_transform(joints[i, 0:2], _trans)
        # # ToTensor & NormalizeTensor
        # if self.transform:
        #     input_img = self.transform(input_img)

        # meta = {
        #     "image": image_file,
        #     "filename": filename,
        #     "joints": joints,
        #     "joints_vis": joints_vis,
        #     "center": center,
        #     "scale": scale,
        #     "rotation": rotation,
        #     "score": score,
        #     "bbox": bbox,
        #     "bbox_id": bbox_id,
        #     "flip_pairs": self.flip_pairs,
        # }
        # 
        # if self.is_target_heatmap:
        #     heatmap, heatmap_weight = self._generate_heatmap(joints, joints_vis)
        #     heatmap = torch.from_numpy(heatmap)
        #     heatmap_weight = torch.from_numpy(heatmap_weight)
        #     if self.is_target_keypoints:
        #         target_joints = torch.from_numpy(joints)
        #         target_joints_vis = torch.from_numpy(joints_vis[:, 1])
        #         return (
        #             input_img,
        #             target_joints,
        #             target_joints_vis,
        #             heatmap,
        #             heatmap_weight,
        #             meta,
        #         )
        #     return input_img, heatmap, heatmap_weight, meta
        # return input_img, target_joints, target_joints_vis, meta
        # ------------------------------------------------------------------------------------------
        
        # TopDownAffine ----------------------------------------
        input_img_mr = []
        joints_mr = []
        
        # joints_single = joints.copy()
        # for j, imageSize in enumerate(self.image_size):
        for imageSize in self.image_size:
            joints_single = np.zeros_like(joints)
            if self.use_udp:
                trans = get_warp_matrix(rotation, center * 2.0, imageSize - 1.0, scale * 200.0)
                input_img = cv2.warpAffine(
                    data_numpy,
                    trans,
                    (int(imageSize[0]), int(imageSize[1])),
                    flags=cv2.INTER_LINEAR,
                )
                joints_single[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), trans)
            else:
                trans = get_affine_transform(center, scale, rotation, imageSize)
                input_img = cv2.warpAffine(
                    data_numpy,
                    trans,
                    (int(imageSize[0]), int(imageSize[1])),
                    flags=cv2.INTER_LINEAR,
                )
                for i in range(self.num_joints):
                    if joints_vis[i, 0] > 0.0:
                        joints_single[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        
        # for _image_size in self.image_size:
        #     input_img, joints_ = self.top_down_affine(data_numpy, _image_size, joints, joints_vis,
        #                                               center, scale, rotation, use_udp=self.use_udp)
            
            # ToTensor & NormalizeTensor ----------------------------------------
            if self.transform:
                input_img = self.transform(input_img)
            
            input_img_mr.append(input_img)
            joints_mr.append(joints_single)
            # joints_mr.append(joints_)
        
        if len(input_img_mr) == 1:
            input_img_mr = input_img_mr[0]
        # if len(joints_mr) == 1:
        #     joints_mr = joints_mr[0]

        # heatmap, heatmap_weight, target_joints, target_joints_vis = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        
        meta = {
            "image": image_file,
            "filename": filename,
            "joints": joints_mr,
            "joints_vis": joints_vis,
            "center": center,
            "scale": scale,
            "rotation": rotation,
            "score": score,
            "bbox": bbox,
            "bbox_id": bbox_id,
            "flip_pairs": self.flip_pairs,
        }

        if isinstance(joints_mr, list):
            target_joints = [torch.from_numpy(joints_) for joints_ in joints_mr]
        else:
            target_joints = torch.from_numpy(joints_mr)
        if len(target_joints) == 1:
            target_joints = target_joints[0]
        target_joints_vis = torch.from_numpy(joints_vis[:, 1])

        if self.is_target_heatmap:
            heatmaps = []
            heatmap_weights = []
            for idx, joint in enumerate(joints_mr):
                heatmap, heatmap_weight = self._generate_heatmap_mr(idx, joint, joints_vis)
                # heatmap, heatmap_weight = self._msra_generate_heatmap(joint, joints_vis, self.image_size[idx],
                #                                                       self.heatmap_size[idx], self.heatmap_type)
                heatmap = torch.from_numpy(heatmap)
                heatmap_weight = torch.from_numpy(heatmap_weight)
                heatmaps.append(heatmap)
                heatmap_weights.append(heatmap_weight)
            
            # if isinstance(heatmaps, list):
            #     heatmaps = [torch.from_numpy(heatmap_) for heatmap_ in heatmaps]
            # else:
            #     heatmaps = torch.from_numpy(heatmaps)
            if len(heatmaps) == 1:
                heatmaps = heatmaps[0] # heatmap개수가 1개면 그대로 반환하고 여러개면 list로 반환
            
            # heatmap_weight = torch.from_numpy(heatmap_weight)
            # print(f"[final] heatmap_weight => {heatmap_weight}")
            if self.is_target_keypoints:
                return (
                    input_img_mr,
                    target_joints,
                    target_joints_vis,
                    heatmaps,
                    heatmap_weights,
                    # heatmap_weight,
                    meta,
                )
            return input_img_mr, heatmaps, heatmap_weight, meta
        return input_img_mr, target_joints, target_joints_vis, meta

    def _generate_heatmap_mr(self, idx, joints, joints_vis):
        """
        :param idx:         joint의 index
        :param joints:      [num_joints, 3]
        :param joints_vis:  [num_joints, 3]
        :return:            heatmap, heatmap_weight(1: visible, 0: invisible)
        """
        
        heatmapSize = self.heatmap_size[idx]
        imageSize = self.image_size[idx]
        
        # heatmap_weight.shape => (17, 1)
        heatmap_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        
        # heatmap_weight는 joints_vis의 x 정보이다. (보이는지 안보이는지)
        heatmap_weight[:, 0] = joints_vis[:, 0]

        assert self.heatmap_type == "gaussian", "Only support gaussian map now!"

        # gaussian heatmap만 지원함.
        if self.heatmap_type == "gaussian":
            # heatmap.shape => (17, 64, 48)
            heatmap = np.zeros(
                (self.num_joints, heatmapSize[1], heatmapSize[0]),
                dtype=np.float32,
            )

            # self.sigma = 2 # tmp_size는 heatmap의 지름...?
            tmp_size = self.sigma * 3
            
            # 모든 joints에 대해서 반복 (17번)
            for joint_id in range(self.num_joints):
                # image와 heatmap의 비율 => (256, 192) / (64, 48) = (4, 3)
                feat_stride = imageSize / heatmapSize
                
                # mu_x, mu_y는 원래 keypoint의 좌표를 heatmap 좌표로 이동시킴.
                try:
                    mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5) # x값을 비율에 맞게 축소 / 0.5 더하는 것은 올림의 의미
                except IndexError as e:
                    print(f"IndexError at idx: {idx}, joint_id: {joint_id}, joints: {joints}")
                    raise e
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5) # y값을 비율에 맞게 축소 / 0.5 더하는 것은 올림의 의미
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # up-left, heatmap의 중심을 기준으로 왼쪽 아래...
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] # bottom-right, heatmap의 중심을 기준으로 오른쪽 위...
                if (
                    ul[0] >= heatmapSize[0]
                    or ul[1] >= heatmapSize[1]
                    or br[0] < 0
                    or br[1] < 0
                ):
                    # If not, just return the image as is
                    heatmap_weight[joint_id] = 0 # joint가 heatmap의 bound를 벗어나면 해당 heatmap의 weight를 0으로 저장.
                    continue

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32) # [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]
                y = x[:, np.newaxis] # y.shape => (13, 1)
                x0 = y0 = size // 2 # 6
                
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmapSize[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmapSize[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmapSize[0])
                img_y = max(0, ul[1]), min(br[1], heatmapSize[1])

                v = heatmap_weight[joint_id]
                if v > 0.5:
                    heatmap[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
                        g_y[0] : g_y[1], g_x[0] : g_x[1]
                    ]
        return heatmap, heatmap_weight
    
    # 현재는 msra style. # udp 적용 x
    def _msra_generate_heatmap(self,
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
        
        # self.heatmap_size
        # self.image_size
        # self.heatmap_type
        # -------------------
        # self.sigma
        # self.num_joints

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

    def top_down_affine(self,
                        img: np.ndarray,
                        image_size: list[int, int],
                        joints: np.ndarray,
                        joints_vis: np.ndarray,
                        center,
                        scale,
                        rotation,
                        use_udp=True) -> tuple[np.ndarray, np.ndarray]:
        """Affine transform the image to make input.
        If use_udp is True, we use udp.

        Args:
            img (np.ndarray): numpy version of original image
            image_size (list[int, int]): Required image size that we want to transform original size to
            joints (np.ndarray): original joints coordinate.
            joints_vis (np.ndarray): joints visible or invisible (visible: 1, invisible: 0)
            center (_type_): center of original image
            scale (_type_): _description_
            rotation (_type_): _description_
            use_udp (bool, optional): _description_. Defaults to True.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                input_img: transformed img with required image size.
                joints_out: transformed joints with required image size.
        """
        joints_out = np.zeros_like(joints)
        if use_udp:
            # 이 matrix를 만들 때, image size를 고려해서 변환 행렬이 만들어짐.
            # 이 matrix는 원본 이미지에서 원하는 image size로 변환해주는 행렬이다.
            # get_warp_matrix 쓰는 것이 udp 적용하는 것이다.
            trans = get_warp_matrix(rotation, center * 2.0, image_size - 1.0, scale * 200.0)
            input_img = cv2.warpAffine(
                img,
                trans,
                (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
            joints_out[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(center, scale, rotation, image_size)
            input_img = cv2.warpAffine(
                img,
                trans,
                (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints_out[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        
        return input_img, joints_out