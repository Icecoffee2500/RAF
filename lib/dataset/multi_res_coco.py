from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
import torch
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from lib.dataset.coco import COCODataset
from lib.dataset.joints_dataset import JointsDataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from timm.data import create_transform
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image


logger = logging.getLogger(__name__)


class MultiResCOCO(COCODataset):
    """_summary_

    Args:
        COCODataset (_type_): _description_
    Description:
        COCODataset을 상속받아서 MultiRes로 데이터를 만들어서 내보낸다.
    """
    def __init__(self, cfg, root, image_set, is_train, transform=None, post_res_transforms=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        
        self.post_transforms = post_res_transforms
    
    def __getitem__(self, idx):
        output = super().__getitem__(idx)

        # if self.is_target_heatmap:
        #     if self.is_target_keypoints:
        #         input_img, target_joints, target_joints_vis, heatmap, heatmap_weight, meta = output
        #     else:
        #         input_img, heatmap, heatmap_weight, meta = output
        # else:
        #     input_img, target_joints, target_joints_vis, meta = output
        
        # # torch.Tensor 또는 np.ndarray 타입을 PIL.Image로 변환
        # if isinstance(input_img, torch.Tensor):
        #     input_img = to_pil_image(input_img)
        # if isinstance(input_img, np.ndarray):
        #     input_img = Image.fromarray(input_img)
        
        # trans_input = []
        # if self.post_transforms is not None:
        #     for tf in self.post_transforms:
        #         input_tf = tf(input_img)
        #         trans_input.append(input_tf)
        # else:
        #     trans_input = [input_img]
        
        # if self.is_target_heatmap:
        #     if self.is_target_keypoints:
        #         return (trans_input, target_joints, target_joints_vis, heatmap, heatmap_weight, meta)
        #     return trans_input, heatmap, heatmap_weight, meta
        # return trans_input, target_joints, target_joints_vis, meta
        return output
    