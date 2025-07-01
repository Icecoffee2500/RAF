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


#Implements the knowledge distillation loss
class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, distillation_type: str, tau: float):
        super().__init__()
        # self.base_criterion = base_criterion # torch.nn.CrossEntropyLoss()
        self.distillation_type = distillation_type # 'smooth-l1'
        # self.alpha = alpha
        self.tau = tau


    def forward(self, outputs_kd=None, teacher_outputs = None):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        # base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'None' or teacher_outputs is None:
            return 0
        
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=False #?
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
            # loss = base_loss  + distillation_loss * self.alpha
        
        elif self.distillation_type == 'cosine':
            distillation_loss = 1 - F.cosine_similarity(outputs_kd, teacher_outputs, eps=1e-6, dim = -1).mean()
            # loss = base_loss  + distillation_loss * self.alpha

        elif self.distillation_type == 'smooth-l1':
            teacher_outputs = F.layer_norm(teacher_outputs, tuple((teacher_outputs.shape[-1],)), eps = 1e-6) # feature whitening 
            distillation_loss = F.smooth_l1_loss(outputs_kd, teacher_outputs)
            # loss = base_loss  + distillation_loss * self.alpha
                
        elif self.distillation_type == 'l2':
            distillation_loss = F.mse_loss(outputs_kd, teacher_outputs)
            # loss = base_loss  + distillation_loss * self.alpha
        else:
            distillation_loss = 0
        
        # loss = distillation_loss * self.alpha
        # loss = distillation_loss

        return distillation_loss