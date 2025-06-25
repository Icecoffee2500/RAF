# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn


class TopdownHeatmapBaseHead(nn.Module):
    """Base class for top-down heatmap heads.

    All top-down heatmap heads should subclass it.
    All subclass should overwrite:

    Methods:`get_loss`, supporting to calculate loss.
    Methods:`get_accuracy`, supporting to calculate accuracy.
    Methods:`forward`, supporting to forward model.
    Methods:`inference_model`, supporting to inference model.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, **kwargs):
        """Forward function."""

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

        return deconv_kernel, padding, output_padding
