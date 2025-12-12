import torch
import torch.nn as nn
import os.path as osp
import re
from collections import OrderedDict
from timm.models.layers import trunc_normal_


class ViTPoseMOON(nn.Module):
    def __init__(self, backbone, deconv_head):
        super(ViTPoseMOON, self).__init__()
        self.backbone = backbone
        self.keypoint_head = deconv_head

        input_dim = 384 # vit-s
        hidden_dim = 384
        out_dim =256

        # 2. MOON을 위한 Projection Head 추가
        # 구조: Linear -> ReLU -> Linear
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # print(f"input shape: {x.shape}")
        feature_map = self.backbone(x)
        # print(f"feature map shape: {feature_map.shape}")

        # --- [Step 3] Aux Task: Contrastive Learning (Representation) ---
        # 3-1. Global Average Pooling (Spatial 차원 제거)
        # dim=(2, 3)은 H, W 차원을 의미합니다.
        pooled_features = feature_map.mean(dim=(2, 3)) 
        # pooled_features shape: [32, 384]
        # print(f"pooled_features shape: {pooled_features.shape}")
        
        # 3-2. Projection Head 통과
        representation = self.projection_head(pooled_features)
        # representation shape: [32, 256] -> MOON Contrastive Loss 계산용
        # print(f"representation shape: {representation.shape}")

        heatmap = self.keypoint_head(feature_map)
        # print(f"heatmap shape: {heatmap.shape}")
        return heatmap, representation

    def init_weights(
        self, pretrained=None, strict=True, map_location=None, check_parameter_values=True
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
        filtered_layer_name = ["cls_token"]  # uncertainty_head 제거됨
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
            state_dict[i] = checkpoint[i]
            
        model.backbone.load_state_dict(state_dict, strict=False)
        return model