# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.utils import resize, normal_init, constant_init
from lib.utils.post_processing import flip_back
from lib.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead


class TopdownHeatmapSimpleHead(TopdownHeatmapBaseHead):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=None,
        in_index=0,
        input_transform=None,
        align_corners=False,
        train_cfg=None,
        test_cfg=None,
        upsample=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.upsample = upsample

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get("target_type", "GaussianHeatmap")

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError("extra should be dict or None.")

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(f"num_deconv_layers ({num_deconv_layers}) should >= 0.")

        identity_final_layer = False
        if extra is not None and "final_conv_kernel" in extra:
            assert extra["final_conv_kernel"] in [0, 1, 3]
            if extra["final_conv_kernel"] == 3:
                padding = 1
            elif extra["final_conv_kernel"] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra["final_conv_kernel"]
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get("num_conv_layers", 0)
                num_conv_kernels = extra.get("num_conv_kernels", [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2,
                        )
                    )
                    layers.append(nn.BatchNorm2d(conv_channels))
                    layers.append(nn.ReLU(inplace=False))
                    # layers.append(nn.ReLU(inplace=True))

            layers.append(
                nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x) # 사실상 얘는 아무것도 안함. (backbone에서 여러개가 나왔을 때를 대비한 것 같음. (아마 ViTPose++용?))
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(), flip_pairs, target_type=self.target_type
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # if self.test_cfg.get("shift_heatmap", False):
                # output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        if not isinstance(inputs, list):
            # if not isinstance(inputs, list):
            if self.upsample > 0:
                inputs = resize(
                    # input=F.relu(inputs),
                    input=F.relu(inputs, inplace=False),
                    scale_factor=self.upsample,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            return inputs

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f"num_layers({num_layers}) " f"!= length of num_filters({len(num_filters)})"
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f"num_layers({num_layers}) " f"!= length of num_kernels({len(num_kernels)})"
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            # layers.append(nn.SyncBatchNorm(planes))
            layers.append(nn.ReLU(inplace=False))
            # layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
