# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight, count=None, wdb=None):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred * target_weight[:, idx], heatmap_gt * target_weight[:, idx]
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        loss = loss / num_joints * self.loss_weight

        return loss

class JointsKLDLoss(nn.Module):
    def __init__(self):
        super(JointsKLDLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        width = output.size(2)
        height = output.size(3)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        # [B, 4096]
        loss = 0.0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            heatmap_pred = F.log_softmax(heatmap_pred.mul(target_weight[:, idx]), dim=1)
            heatmap_gt = F.softmax(heatmap_gt.mul(target_weight[:, idx]), dim=1)

            loss += self.criterion(
                heatmap_pred, heatmap_gt
            )

        loss = torch.sum(torch.sum(loss, dim=1), dim=0)

        return loss / batch_size / (width * height)