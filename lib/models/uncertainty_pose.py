import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import re
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from copy import deepcopy


class UncertaintyPose(nn.Module):
    def __init__(self, backbone, keypoint_head, uncertainty_head, config, device=None):
        super(UncertaintyPose, self).__init__()

        self.scale = config.MODEL.SCALE
        self.use_head = config.MODEL.USE_AFTER_KP_HEAD
        self.sum_to_1 = config.MODEL.SUM_TO_ONE

        self.backbone = backbone
        self.keypoint_head = keypoint_head
        self.uncertainty_head = uncertainty_head

        self.heatmap_size = config.MODEL.HEATMAP_SIZE
        self.x = torch.tensor([i for i in range(1, self.heatmap_size[0] + 1)])
        self.y = torch.tensor([i for i in range(1, self.heatmap_size[1] + 1)])
        # if device:
        #     self.x = self.x.to(device)
        #     self.y = self.y.to(device)
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing="xy")

        self.act1 = nn.ReLU()

    def forward(self, x):
        feature_map, output_kd = self.backbone(x)
        heatmap = self.keypoint_head(feature_map)

        normalized_heatmap = heatmap

        if self.sum_to_1:
            normalized_heatmap = torch.clip(normalized_heatmap, 0, 10) + 1e-8 # 값을 0~10 사이로 제한함. NaN 방지를 위해서 1e-8 더해줌.
            sum_hm = (
                torch.sum(normalized_heatmap, dim=(2, 3)) # 2번째와 3번째 차원 (64, 48) 부분을 따라서 합을 계산함. # (B, 17)
                .unsqueeze(2) # 차원 늘려주기 (B, 17, 1)
                .unsqueeze(2) # 차원 늘려주기 (B, 17, 1, 1)
                .repeat(1, 1, self.heatmap_size[1], self.heatmap_size[0]) # shape을 원래 크기로 확장함. (B, 17, 64, 48)
            )
            normalized_heatmap = normalized_heatmap / (sum_hm + 1e-8) # 모든 값들이 0~1 사이로 정규화된다.

        # Scale up
        if self.scale != 1:
            normalized_heatmap = torch.clamp(normalized_heatmap, min=-0.5, max=0.5)
            normalized_heatmap = normalized_heatmap * self.scale * 2 - self.scale

        self.xx = self.xx.to(feature_map.device)
        self.yy = self.yy.to(feature_map.device)
        expected_keypoints, normalized_heatmap = self._get_expected_keypoints(normalized_heatmap)
        # expected_keypoints, heatmap = self._get_expected_keypoints_using_original_heatmap(heatmap)
        uncertainty = self.uncertainty_head(feature_map)

        return expected_keypoints, uncertainty, normalized_heatmap, heatmap
        # return expected_keypoints, uncertainty, heatmap

    def _get_expected_keypoints(self, heatmap):
        B, C, H, W = heatmap.shape
        normalized_heatmap = heatmap.reshape((B, C, -1))

        if not self.sum_to_1:
            normalized_heatmap = torch.softmax(normalized_heatmap, dim=-1)

        normalized_heatmap = normalized_heatmap.reshape((B, C, H, W))
        
        expected_keypoints_x = torch.sum(torch.mul(self.xx, normalized_heatmap), dim=(2, 3)) # heatmap을 x축에 projection 시킨 후에, 그 값들의 기댓값을 취하여 expected keypoint의 x좌표를 얻어냄.
        expected_keypoints_y = torch.sum(torch.mul(self.yy, normalized_heatmap), dim=(2, 3))
        expected_keypoints = torch.stack([expected_keypoints_x, expected_keypoints_y], dim=-1) - 1
        expected_keypoints = expected_keypoints
        return expected_keypoints, normalized_heatmap

    # def _get_expected_keypoints_using_original_heatmap(self, heatmap):
    #     # heatmap = self.act1(heatmap) + 1e-4
    #     expected_keypoints_x = torch.sum(torch.mul(self.xx, heatmap), dim=(2, 3))
    #     expected_keypoints_y = torch.sum(torch.mul(self.yy, heatmap), dim=(2, 3))
    #     expected_keypoints = torch.stack([expected_keypoints_x, expected_keypoints_y], dim=-1) - 1
    #     expected_keypoints = expected_keypoints
    #     return expected_keypoints, heatmap

    @staticmethod
    def get_max_preds(heatmaps):
        """Get keypoint predictions from score maps.
        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W
        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        Returns:
            tuple: A tuple containing aggregated results.
            - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
        assert heatmaps.ndim == 4, "batch_images should be 4-ndim"

        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, 0) * 4
        return preds, maxvals

    def init_weights(
        self,
        pretrained=None,
        strict=True,
        map_location=None,
        check_parameter_values=True,
    ):
        if isinstance(pretrained, str):
            self.checkpoint = self._load_from_local(pretrained, map_location)
            try:
                self.load_state_dict(self.checkpoint["state_dict"], strict=strict)
            except RuntimeError as e:
                print("Please verify once again!!")
                errors = re.split(",", str(e))
                for idx, e in enumerate(errors):
                    if idx == 0 or idx == len(errors) - 1:
                        continue
                    print(f"Not applied {e.strip()}")
            finally:
                print("Succesfully init weights..")
        elif pretrained is None:

            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)
        else:
            raise TypeError(
                "pretrained must be a str or None." f" But received {type(pretrained)}."
            )

        if check_parameter_values:
            org_checkpoint = torch.load(pretrained)["state_dict"]
            checkpoint_state_dict = self.checkpoint["state_dict"]
            for key in checkpoint_state_dict.keys():
                if isinstance(checkpoint_state_dict[key], torch.nn.Parameter):
                    # for parameters, compare the data attribute
                    assert torch.equal(
                        checkpoint_state_dict[key].data, org_checkpoint[key].data
                    ), f"Parameter {key} is different between checkpoint and model!"
                else:
                    # assert torch.equal(
                    #     checkpoint_state_dict[key], org_checkpoint[key]
                    # ), f"Layer {key} is different between checkpoint and model!"
                    pass
            print("The parameters of the original pretrained model and your model are identical!")

    @staticmethod
    def _load_from_local(filename, map_location=None):
        if not osp.isfile(filename):
            raise IOError(f"{filename} is not a checkpoint file")
        checkpoint = torch.load(filename, map_location=map_location)

        changed_checkpoint = {}
        changed_model = OrderedDict()
        filtered_layer_name = ["cls_token", "uncertainty_head"]
        for state_dict_name, model in checkpoint.items():
            for layer, params in model.items():
                layer_without_backbone_name = layer
                if any(s in layer_without_backbone_name for s in filtered_layer_name):
                    continue
                if layer_without_backbone_name in filtered_layer_name:
                    continue
                changed_model[layer_without_backbone_name] = params
            changed_checkpoint[state_dict_name] = changed_model

        return changed_checkpoint


    @staticmethod
    def custom_init_weights(model, checkpoint_path):
        print("checkpoint path : ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        try:
            c_keys = list(checkpoint["state_dict"].keys())
            c_sd = checkpoint["state_dict"]
        except:
            c_sd = checkpoint
            c_keys = list(checkpoint.keys())

        m_sd = model.state_dict()
        m_keys = list(model.state_dict().keys())

        for i in range(len(m_keys)):
            try:
                if c_sd[c_keys[i]].shape != m_sd[m_keys[i]].shape:
                    print("Please verify once again!! >>", end=" ")
                    print(c_keys[i], m_keys[i])
                if c_sd[c_keys[i]].shape == m_sd[m_keys[i]].shape:
                    m_sd[m_keys[i]] = c_sd[c_keys[i]]
            except IndexError:
                print("index is over :", m_keys[i])

        print("Succesfully init weights..")
        model.load_state_dict(m_sd, strict=False)
        return model

    def vit_mae_init(self, model, checkpoint_path, device='cpu'):
        # checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]
        checkpoint = torch.load(checkpoint_path, map_location=device)['model']

        # model.load_state_dict(checkpoint, strict=False)

        state_dict = model.backbone.state_dict()

        checkpoint_keys = list(checkpoint.keys())

        for i in checkpoint_keys:
            if "cls_token" in i:
                checkpoint.pop(i, None)

        checkpoint_keys = list(checkpoint.keys())

        for i in checkpoint_keys:
            if "pos_embed" in i:
                state_dict[i] = checkpoint[i][:,:193]
                continue
            elif "patch_embed" in i:
                if state_dict[i].shape == checkpoint[i].shape:
                    state_dict[i] = checkpoint[i]
                continue
            state_dict[i] = checkpoint[i]


        model.backbone.load_state_dict(state_dict, strict=False)
        return model