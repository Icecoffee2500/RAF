import numpy as np
import torchvision.transforms as transforms
from collections.abc import Sequence
from typing import Union, Type
import torch

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