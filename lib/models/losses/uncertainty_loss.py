import torch
import torch.nn as nn


class UncertaintyExpMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(UncertaintyExpMSELoss, self).__init__()
        self.epsilon = 1e-5
        self.use_target_weight = use_target_weight

    def forward(self, pred_keypoints, target_keypoints, target_weights, uncertainties):
        B, N, _ = pred_keypoints.shape
        pred_keypoints = pred_keypoints.reshape((B, N, -1)).split(1, 1)
        target_keypoints = target_keypoints.reshape((B, N, -1)).split(1, 1)
        uncertainties = uncertainties.reshape((B, N, -1)).split(1, 1)

        loss1 = 0
        loss2 = 0
        for idx in range(N):
            pred_keypoint = pred_keypoints[idx].squeeze()
            target_keypoint = target_keypoints[idx].squeeze()
            log_simga2 = uncertainties[idx].squeeze()
            target_weight = target_weights[:, idx].repeat(2).view(2, -1).transpose(0, 1)
            if self.use_target_weight:
                squared_error = (
                    target_keypoint.mul(target_weight) - pred_keypoint.mul(target_weight)
                ) ** 2
                loss1 += torch.sum(torch.exp(-log_simga2) * squared_error)
                loss2 += torch.sum(log_simga2)

        loss = 0.25 * (loss1 + loss2) / B / N
        return loss


class UncertaintySigMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(UncertaintySigMSELoss, self).__init__()
        self.epsilon = 1e-5
        self.use_target_weight = use_target_weight

    def forward(self, keys, uncertainty, target_keys, target_weight):
        batch_size = keys.size(0)
        num_joints = keys.size(1)

        expected_keys = keys.reshape((batch_size, num_joints, -1)).split(1, 1)
        gt_keys = target_keys.reshape((batch_size, num_joints, -1)).split(1, 1)
        uncertainty_pred = uncertainty.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0
        loss_log = 0
        for idx in range(num_joints):
            expected_key = expected_keys[idx].squeeze()
            gt_key = gt_keys[idx].squeeze()
            uncertainty_pred_ = uncertainty_pred[idx].squeeze()

            joint_weight = target_weight[:, idx].repeat(2).view(2, -1).transpose(0, 1)

            loss += 0.25 * torch.sum(
                (
                    (gt_key.mul(joint_weight) - expected_key.mul(joint_weight))
                    / (1 + uncertainty_pred_ + self.epsilon)
                )
                ** 2
            )

            # loss += 0.25* \
            #             torch.sum(((gt_key.mul(joint_weight) \
            #                    - expected_key.mul(joint_weight)) \
            #                       / (uncertainty_pred_+ self.epsilon))**2)

            loss_log += torch.sum(torch.log(1 + torch.abs(uncertainty_pred_.mul(joint_weight))))

            # loss_log += torch.sum(torch.log(
            #                 torch.abs(uncertainty_pred_.mul(joint_weight))))

        return (loss + loss_log) / num_joints / batch_size


class UncertaintySoftPlusMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(UncertaintySoftPlusMSELoss, self).__init__()
        self.epsilon = 1e-5
        self.use_target_weight = use_target_weight

        self.soft_plus = torch.nn.Softplus()

    def forward(self, keys, target_keys, target_weight, uncertainty, count=None, wdb=None):
        # keys => (B, 17, 2)
        # target_keys => (B, 17, 2)
        # target_weight => (B, 17, 1)
        # uncertainty => (B, 17, 2)
        batch_size = keys.size(0)
        num_joints = keys.size(1)
        expected_keys = keys.reshape((batch_size, num_joints, -1)).split(1, 1) # num_joints 개수 만큼의 원소를 가진 리스트 => (B, 1, 2) x 17개
        gt_keys = target_keys.reshape((batch_size, num_joints, -1)).split(1, 1)

        uncertainty = self.soft_plus(uncertainty)

        uncertainty_pred = uncertainty.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        loss_log = 0
        
        # 각 joint에 대해서 loss를 계산해서 합친다.
        for idx in range(num_joints):
            expected_key = expected_keys[idx].squeeze() # (B, 2)
            gt_key = gt_keys[idx].squeeze() # (B, 2)

            uncertainty_pred_ = uncertainty_pred[idx].squeeze() # (B, 2)
            joint_weight = target_weight[:, idx].repeat(2).view(2, -1).transpose(0, 1) # (B, 2)
            loss += 0.25 * torch.sum(
                (
                    (gt_key.mul(joint_weight) - expected_key.mul(joint_weight))
                    / (uncertainty_pred_ + self.epsilon)
                )
                ** 2
            )

            loss_log += torch.sum(torch.log((uncertainty_pred_.mul(joint_weight) + 1e-12) ** 2))

        loss = loss / num_joints / batch_size
        loss_log = loss_log / num_joints / batch_size / 2

        if count and (count % 200 == 0):
            print(f"Uncertatiny MSE loss[{loss}] UNC Sigma loss[{loss_log}]")

            if wdb:
                wdb.log({"UNC MSE loss": loss})
                wdb.log({"UNC Sigma loss": loss_log})

        return (loss + loss_log) / 2 
