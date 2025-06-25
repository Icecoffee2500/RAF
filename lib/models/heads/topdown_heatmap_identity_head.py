import numpy as np
import torch.nn as nn

from lib.utils.post_processing import flip_back


class TopdownHeatmapIdentityHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channls,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get("target_type", "GaussianHeatmap")

        self.final_layer = nn.Conv2d(in_channels, out_channls, 1, stride=1)

    def forward(self, x):
        """Forward function."""

        # return heatmap
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
            if self.test_cfg.get("shift_heatmap", False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap
