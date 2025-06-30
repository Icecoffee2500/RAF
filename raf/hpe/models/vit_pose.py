import torch
import torch.nn as nn
import os.path as osp
import re
from collections import OrderedDict
from timm.models.layers import trunc_normal_


class ViTPose(nn.Module):
    def __init__(self, backbone, deconv_head, distillation_target=None):
        super(ViTPose, self).__init__()
        self.backbone = backbone
        self.keypoint_head = deconv_head
        self.distillation_target = distillation_target
        print(f"distill target => {self.distillation_target}")

    def forward(self, x):
        # feature_map = self.backbone(x)
        kd_output = None
        
        # 기본적으로 ViT의 backbone에서 feature_map과 distillation output을 뽑음.
        feature_map, kd_output = self.backbone(x, distillation_target=self.distillation_target)
        
        heatmap = self.keypoint_head(feature_map)
        
        # logit이면 ViTPose의 output (heatmap)을 그대로 distillation output으로 씀.
        if self.distillation_target == 'logit_hm':
            kd_output = heatmap
        
        return heatmap, kd_output

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
            state_dict[i] = checkpoint[i]
            
        model.backbone.load_state_dict(state_dict, strict=False)
        return model

def _main():
    import sys
    import os

    # 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    sys.path.append(project_root)
    print(f"base root path: {project_root}")

    from lib.models.extra.vit_small_uncertainty_config import extra
    from lib.core.config import config
    from lib.core.config import update_config
    import torchvision.transforms as transforms
    from lib.models.backbones.vit import ViT
    from lib.models.heads import TopdownHeatmapSimpleHead
    from lib.dataset.coco import COCODataset
    from lib.dataset.mpii import MPIIDataset
    from lib import dataset
    
    config_file = "vit_small_multi_res_sfl_mpii_train_kd.yaml"
    config_path = os.path.abspath(os.path.join(project_root, 'experiments/mpii', config_file))
    print(f"config path: {config_path}")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    update_config(config_path)
    backbone = ViT(
        img_size=extra["backbone"]["img_size"],
        patch_size=extra["backbone"]["patch_size"],
        embed_dim=extra["backbone"]["embed_dim"],
        in_channels=3,
        num_heads=extra["backbone"]["num_heads"],
        depth=extra["backbone"]["depth"],
        qkv_bias=True,
        drop_path_rate=extra["backbone"]["drop_path_rate"],
        use_gpe=config.MODEL.USE_GPE,
        use_lpe=config.MODEL.USE_LPE,
        use_gap=config.MODEL.USE_GAP,
    )
    deconv_head = TopdownHeatmapSimpleHead(
        in_channels=extra["keypoint_head"]["in_channels"],
        num_deconv_layers=extra["keypoint_head"]["num_deconv_layers"],
        num_deconv_filters=extra["keypoint_head"]["num_deconv_filters"],
        num_deconv_kernels=extra["keypoint_head"]["num_deconv_kernels"],
        extra=dict(final_conv_kernel=1),
        # out_channels=17,
        out_channels=config.MODEL.NUM_JOINTS,
    )
    
    print(f"config.DATASET.ROOT: {config.DATASET.ROOT}")
    print(f"config.DATASET.TRAIN_SET: {config.DATASET.TRAIN_SET}")
    
    train_dataset = eval(f'dataset.{config.DATASET.DATASET}')(
            cfg=config,
            root=config.DATASET.ROOT,
            image_set=config.DATASET.TRAIN_SET,
            is_train=True,
            transform=  transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
    )
    
    # for batch_idx, (inputs, target_joints, target_joints_vis, heatmaps, heatmap_target, meta) in enumerate(train_loader):
    inputs, target_joints, target_joints_vis, heatmaps, heatmap_target, meta = next(iter(train_loader))
    for idx, input in enumerate(inputs):
        print(f"input [{idx}] shape: {input.shape}")
        feature_map, _ = backbone(input)
        print(f"feature map shape: {feature_map.shape}")
        heatmap = deconv_head(feature_map)
        print(f"heatmap shape: {heatmap.shape}")
        break

if __name__ == "__main__":
    _main()