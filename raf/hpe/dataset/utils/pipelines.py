# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
import random
import torchvision.transforms as transforms

from hpe.dataset.utils.transforms import (
    fliplr_joints, get_affine_transform, affine_transform,
    get_warp_matrix, warp_affine_joints
)
from common.io import imfrombytes

class LoadImageFromFile:
    """Loading image(s) from file.
    
    Args:
        image_file (str): image file path
    Returns:
        img (np.ndarray): original numpy image
    """
    def __init__(self, logger):
        self.logger = logger
    
    def _get(self, filepath: str) -> bytes: # common function
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
        img_bytes = self._get(path)
        img = imfrombytes(img_bytes, "color", channel_order="rgb") # bytes -> ndarrays로 바꿔줌. # (H, W, C:rgb)
        
        if img is None:
            raise ValueError(f"Fail to read {path}")
        
        return img
    
    def __call__(self, image_file):
        if image_file is not None:
            img = self._read_image(image_file) # 이미지 numpy (H, W, C) C->'rgb'
            assert isinstance(img, np.ndarray)
        else:
            self.logger.error(f"=> Fail to read {image_file}")
            raise ValueError(f"Fail to read {image_file}")
        
        return img

# TopDowns
class TopDownGetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.

    Args:
        scale (int)

    Returns
        scale (int)
        rotation (float)

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """
    def __init__(self, scale_factor=0.5, rotation_factor=40, rotation_prob=0.6):
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.rotation_prob = rotation_prob
    
    def __call__(self, scale):
        sf = self.scale_factor
        rf = self.rotation_factor
        
        scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        rotation = (
            np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= self.rotation_prob else 0
        )
        
        return scale, rotation

class TopDownRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_2d', 'joints_2d_vis', 'center'

    Modifies key: 'img', 'joints_2d', 'joints_2d_vis', 'center' and 'flipped'.

    Args:
        flip_prob (float): Probability of flip.
    """
    def __init__(self, flip_probs=0.5):
        self.flip_probs = flip_probs
    
    def __call__(self, img, joints, joints_vis, center, flip_pairs):
        """Perform data augmentation with random image flip."""
        
        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        if random.random() <= self.flip_probs:
                
            # 이미지 좌우 뒤집기.
            img = img[:, ::-1, :]
            
            # joints, joints_vis 좌우 뒤집기.
            joints, joints_vis = fliplr_joints(
                joints, joints_vis, img.shape[1], flip_pairs
            )
            
            # center도 좌우 뒤집기.
            center[0] = img.shape[1] - center[0] - 1
        
        return img, joints, joints_vis, center

class TopDownHalfBodyTransform:
    def __init__(self, upper_body_ids, image_size, num_joints_half_body=8, prob_half_body=0.3):
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body
        self.upper_body_ids = upper_body_ids
        self.image_size = image_size
    
    def half_body_transform(self,
                            num_joints,
                            joints: np.ndarray,
                            joints_visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get center&scale for half-body transform.

        Args:
            num_joints (int): the number of joints
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
        for joint_id in range(num_joints):
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
        
        if isinstance(self.image_size[0], (np.ndarray, list)):
            aspect_ratio = self.image_size[0][0] / self.image_size[0][1]
        else:
            aspect_ratio = self.image_size[0] / self.image_size[1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.5
        return center, scale
    
    def __call__(self, num_joints, joints, joints_vis):
        if (
            np.sum(joints_vis[:, 0]) > self.num_joints_half_body
            and np.random.rand() < self.prob_half_body
        ):
            c_half_body, s_half_body = self.half_body_transform(num_joints, joints, joints_vis)
            return c_half_body, s_half_body
        return None, None

class TopDownAffine:
    """Affine transform the image to make input.

    Required keys:'img', 'joints_2d', 'joints_2d_vis', 'ann_info','scale', 'rotation' and 'center'.

    Modified keys:'img', 'joints_2d', and 'joints_2d_vis'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """
    def __init__(self, image_size, use_udp=False):
        self.use_udp = use_udp
        self.image_size = image_size
    
    def __call__(self, img, joints, joints_vis, center, scale, rotation, num_joints):
        """Affine transform the image to make input."""
        # _joints_transformed = np.zeros_like(joints)
        
        if not isinstance(self.image_size[0], (np.ndarray, list)):
            _joints_transformed = np.zeros_like(joints)
            if self.use_udp:
                # 이 matrix를 만들 때, image size를 고려해서 변환 행렬이 만들어짐.
                # 이 matrix는 원본 이미지에서 원하는 image size로 변환해주는 행렬이다.
                trans = get_warp_matrix(rotation, center * 2.0, self.image_size - 1.0, scale * 200.0)
                input_img = cv2.warpAffine(
                    img,
                    trans,
                    (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR,
                )
                _joints_transformed[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), trans)
            else:
                trans = get_affine_transform(center, scale, rotation, self.image_size)
                input_img = cv2.warpAffine(
                    img,
                    trans,
                    (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR,
                )
                for i in range(num_joints):
                    if joints_vis[i, 0] > 0.0:
                        _joints_transformed[i, 0:2] = affine_transform(joints[i, 0:2], trans)
            return input_img, _joints_transformed
        else:
            input_img = []
            joints_transformed = []
            for _image_size in self.image_size:
                _joints_transformed = np.zeros_like(joints)
                if self.use_udp:
                    # 이 matrix를 만들 때, image size를 고려해서 변환 행렬이 만들어짐.
                    # 이 matrix는 원본 이미지에서 원하는 image size로 변환해주는 행렬이다.
                    trans = get_warp_matrix(rotation, center * 2.0, _image_size - 1.0, scale * 200.0)
                    _input_img = cv2.warpAffine(
                        img,
                        trans,
                        (int(_image_size[0]), int(_image_size[1])),
                        flags=cv2.INTER_LINEAR,
                    )
                    _joints_transformed[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), trans)
                else:
                    trans = get_affine_transform(center, scale, rotation, _image_size)
                    _input_img = cv2.warpAffine(
                        img,
                        trans,
                        (int(_image_size[0]), int(_image_size[1])),
                        flags=cv2.INTER_LINEAR,
                    )
                    for i in range(num_joints):
                        if joints_vis[i, 0] > 0.0:
                            _joints_transformed[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                
                input_img.append(_input_img)
                joints_transformed.append(_joints_transformed)
        
            return input_img, joints_transformed

class TopDownGenerateHeatmap:
    def __init__(self, image_size, heatmap_size, sigma, heatmap_type):
        self.sigma = sigma
        self.heatmap_type = heatmap_type
        self.image_size = image_size
        self.heatmap_size = heatmap_size
    
    def _msra_generate_heatmap(
        self,
        joints: np.ndarray,
        joints_vis: np.ndarray,
        image_size: list,
        heatmap_size: list,
        num_joints: int,
        heatmap_type: str = "gaussian")-> tuple[np.ndarray, np.ndarray]:
        """microsoft style heatmap generation

        Args:
            joints (np.ndarray): The position of body joints. (num_joints, 2)
            joints_visible (np.ndarray): The visibility of body joints. (num_joints, 2), visible: 1, invisible: 0.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                heatmap (num_joints, heatmap_height, heatmap_width),
                heatmap_weight (num_joints, 1). visible: 1, invisible: 0.
        """
        heatmap_weight = np.ones((num_joints, 1), dtype=np.float32)
        heatmap_weight[:, 0] = joints_vis[:, 0]

        assert heatmap_type == "gaussian", "Only support gaussian map now!"

        if heatmap_type == "gaussian":
            heatmap = np.zeros(
                (num_joints, heatmap_size[1], heatmap_size[0]),
                dtype=np.float32,
            )

            # tmp_size는 heatmap의 반지름
            # 3-sigma rule # sigma에 3배를 하면 분포의 99.7% (거의 대부분)을 차지함.
            tmp_size = self.sigma * 3
            
            for joint_id in range(num_joints):
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
    
    def __call__(self, joints_transformed, joints_vis, num_joints):
        if not isinstance(self.heatmap_size[0], (np.ndarray, list)):
            heatmaps, heatmap_weights = self._msra_generate_heatmap(
                joints_transformed,
                joints_vis,
                self.image_size,
                self.heatmap_size,
                num_joints,
                self.heatmap_type)
            heatmaps = torch.from_numpy(heatmaps)
            heatmap_weights = torch.from_numpy(heatmap_weights)
        else:
            heatmaps = []
            heatmap_weights = []
            
            for idx, (_image_size, _heatmap_size) in enumerate(zip(self.image_size, self.heatmap_size)):
                _heatmaps, _heatmap_weights = self._msra_generate_heatmap(
                    joints_transformed[idx],
                    joints_vis,
                    _image_size,
                    _heatmap_size,
                    num_joints,
                    self.heatmap_type)
                
                _heatmaps = torch.from_numpy(_heatmaps)
                _heatmap_weights = torch.from_numpy(_heatmap_weights)
                
                heatmaps.append(_heatmaps)
                heatmap_weights.append(_heatmap_weights)
        
        return heatmaps, heatmap_weights

# 그 외.
class RandomShiftBboxCenter:
    """Data augmentation with random shift center of bounding box.

    Required keys: 'center', 'scale'

    Modifies key: 'center', 'scale'

    Args:
        shift_factor (float): Probability of shift.
        pixel_std (float): standard deviation of pixels.
    """
    def __init__(self, shift_factor, shift_prob, pixel_std):
        self.shift_factor = shift_factor
        self.pixel_std = pixel_std
        self.shift_prob = shift_prob
    
    def __call__(self, center, scale):
        if np.random.rand() < self.shift_prob:
            center = center + np.random.uniform(-1, 1, 2) * self.shift_factor * scale * self.pixel_std
        
        return center

class ToTensor:
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, input_img):
        # if isinstance(input_img, (list, np.ndarray)):
        if isinstance(input_img, list):
            input_img = [transforms.ToTensor()(img) for img in input_img]
        else:
            input_img = transforms.ToTensor()(input_img)

        return input_img


class NormalizeTensor:
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, input_img):
        if isinstance(input_img, (list, np.ndarray)):
            input_img = [
                transforms.Normalize(mean=self.mean, std=self.std)(img)
                for img in input_img
            ]
        else:
            input_img = transforms.Normalize(mean=self.mean, std=self.std)(input_img)

        return input_img