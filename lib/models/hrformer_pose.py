import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import re
from collections import OrderedDict
from timm.models.layers import trunc_normal_


class HRFormerPose(nn.Module):
    def __init__(self, backbone, head):
        super(HRFormerPose, self).__init__()
        self.backbone = backbone
        self.keypoint_head = head

    def forward(self, x):
        feature_map = self.backbone(x)
        heatmap = self.keypoint_head(feature_map)
        return heatmap

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

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1) * 4
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
                    assert torch.equal(
                        checkpoint_state_dict[key], org_checkpoint[key]
                    ), f"Layer {key} is different between checkpoint and model!"
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
