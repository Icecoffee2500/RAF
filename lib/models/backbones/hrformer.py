import torch
import numpy as np
import torch.nn as nn
import math
import copy
import os.path as osp

from collections import OrderedDict
from torch.nn.functional import pad
from timm.models.layers import drop_path, trunc_normal_
from lib.utils.utils import build_norm_layer


class Bottleneck_block(nn.Module):
    """
    Bottleneck block used ResNet

    1x1 Conv
    3x3 Conv
    1x1 Conv

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of 'out_channel/mid_channels' where
            'mid_channels' is the input/output channels of conv2, Default: 4.


    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expansion=4,
        dilation=1,
        downsample=None,
        norm_cfg=dict(type="BN"),
    ):
        # deep copy!??!
        # why deep copy here???
        # print("norm_cfg deepcopy")
        norm_cfg = copy.deepcopy(norm_cfg)

        super(Bottleneck_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            dilation=dilation,
            bias=False,
        )

        self.norm1 = build_norm_layer(norm_cfg, self.mid_channels)
        self.conv2 = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding="same",
            dilation=dilation,
            bias=False,
        )
        self.norm2 = build_norm_layer(norm_cfg, self.mid_channels)
        self.conv3 = nn.Conv2d(
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            dilation=dilation,
            bias=False,
        )
        self.norm3 = build_norm_layer(norm_cfg, self.out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


class HRFormer_module(nn.Module):
    def __init__(
        self,
        window_size,
        num_channel,
        num_branch,
        num_blocks,
        num_heads,
        mlp_ratios,
        drop_paths,
        with_rpe=True,
        norm_cfg=dict(type="BN", requires_grad=True),
        transformer_norm_cfg=dict(type="LN", eps=1e-6),
        upsample_cfg=dict(mode="nearest", align_corners=None),
    ):
        super().__init__()
        self.window_size = window_size
        self.num_channel = num_channel
        self.num_branch = num_branch
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.drop_paths = drop_paths
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.transformer_norm_cfg = copy.deepcopy(transformer_norm_cfg)
        self.with_rpe = with_rpe
        self.upsample_cfg = upsample_cfg

        self.branches = self._make_branch()
        # print("Branch : ", len(self.branches))
        self.fuse_layers = self._make_fuse_layers()

        self.relu = nn.ReLU()

    def _make_one_branch(self, idx):
        layers = []
        for i in range(self.num_blocks[idx]):
            layers.append(
                HRFormer_block(
                    self.num_channel[idx],
                    self.num_heads[idx],
                    self.window_size[idx],
                    self.mlp_ratios[idx],
                    self.drop_paths[i],
                    self.norm_cfg,
                    self.transformer_norm_cfg,
                )
            )
            # print(self.drop_paths[i])

        return nn.Sequential(*layers)

    def _make_branch(self):
        branch_layer = []
        for idx in range(self.num_branch):
            branch_layer.append(self._make_one_branch(idx))

        return nn.ModuleList(branch_layer)

    def _make_fuse_layers(self):
        fuse_layers = []
        for i in range(self.num_branch):
            fuse_layer = []
            for j in range(self.num_branch):
                temporal_layer = []
                if i == j:
                    fuse_layer.append(None)
                    continue

                # for downsample
                if self.num_channel[j] < self.num_channel[i]:
                    num_down = i - j
                    # print("iter num : ", num_down)
                    if num_down != 1:
                        for _ in range(1, num_down):
                            temporal_layer += self.downsample_block_in_fuse(
                                self.num_channel[j], self.num_channel[j]
                            )
                            temporal_layer.append(nn.ReLU(inplace=True))

                    temporal_layer += self.downsample_block_in_fuse(
                        self.num_channel[j], self.num_channel[i]
                    )

                # for upsample
                else:
                    temporal_layer.append(
                        nn.Conv2d(
                            self.num_channel[j],
                            self.num_channel[i],
                            kernel_size=1,
                            bias=False,
                        )
                    )
                    temporal_layer.append(build_norm_layer(self.norm_cfg, self.num_channel[i]))
                    temporal_layer.append(
                        nn.Upsample(
                            scale_factor=2 ** (j - i),
                            mode=self.upsample_cfg["mode"],
                            align_corners=self.upsample_cfg["align_corners"],
                        )
                    )
                # print("temporal_layer : ", temporal_layer)
                fuse_layer.append(nn.Sequential(*temporal_layer))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def downsample_block_in_fuse(self, in_channel, out_channel):
        temporal_layer = []
        temporal_layer.append(
            nn.Conv2d(
                in_channel,
                in_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channel,
                bias=False,
            )
        )
        temporal_layer.append(build_norm_layer(self.norm_cfg, in_channel))
        temporal_layer.append(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        )
        temporal_layer.append(build_norm_layer(self.norm_cfg, out_channel))
        return temporal_layer

    def forward(self, x):
        # branch가 1개이면 fuse할 필요 없어서 바로 return
        if self.num_branch == 1:
            return [self.branches[0](x[0])]

        # print("Go to branch ")
        for i in range(self.num_branch):
            x[i] = self.branches[i](x[i])
            # print(f"branch {i} shape : ",x[i].shape)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branch):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            # print(f"fuse_{i} result : ", y.shape)

            # 다른 layer들로부터 특정 branch로 fuse된 값들을 relu로 합쳐줌
            x_fuse.append(self.relu(y))
        return x_fuse


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


class HRFormer_block(nn.Module):
    def __init__(
        self,
        num_channel,
        num_heads,
        window_size,
        mlp_ratio,
        drop_rate,
        norm_cfg,
        transformer_norm_cfg=dict(type="LN", eps=1e-6),
    ):
        super().__init__()
        self.num_channel = num_channel
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.transformer_norm_cfg = copy.deepcopy(transformer_norm_cfg)

        self.norm1 = build_norm_layer(self.transformer_norm_cfg, self.num_channel)
        self.attn = LW_MSA(
            num_channel=num_channel,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
            act_cfg=dict(type="GELU"),
            norm_cfg=self.norm_cfg,
            transformer_norm_cfg=self.transformer_norm_cfg,
        )

        self.norm2 = build_norm_layer(self.transformer_norm_cfg, self.num_channel)
        self.ffn = DW_CNN(
            num_channel=num_channel,
            mlp_ratio=mlp_ratio,
            act_cfg=dict(type="GELU"),
            norm_cfg=self.norm_cfg,
        )

        self.drop_path = DropPath(self.drop_rate) if self.drop_rate > 0.0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()

        # try:
        #     print("HRForemr_block input shape : ", x.shape)
        # except:
        #     print("Error ! ", x)

        # Attention
        # 모든 feature의 dim에 대해 LayerNorm
        x = x.view(B, C, -1).permute(0, 2, 1)
        # Normalized 된 data를 attention layer에 넣음.
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # print("after attn shape : ", x.shape)
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


# https://www.youtube.com/watch?v=vVaRhZXovbw
# Efficient net에서 처음 소개한 방법으로 보임.
class DW_CNN(nn.Module):
    def __init__(
        self,
        num_channel,
        mlp_ratio,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="SyncBN", eps=1e-5),
        dw_act_cfg=dict(type="GELU"),
    ):
        super().__init__()
        norm_cfg = copy.deepcopy(norm_cfg)

        hidden_channel = num_channel * mlp_ratio

        self.conv1 = nn.Conv2d(num_channel, hidden_channel, kernel_size=1)
        self.act1 = nn.GELU()
        self.norm1 = build_norm_layer(norm_cfg, hidden_channel)

        # group을 사용하면 Conv layer 계산에 필요한 parameter 수를 줄일 뿐 아니라,
        # 다양한 필터가 입력 채널의 다양한 subset을 통과 시키도록하여 다양성을 촉진한다.?!?!

        # group을 필터 개수만큼 하면 이것이 depth-wise CNN임.
        self.DWConv = nn.Conv2d(
            hidden_channel,
            hidden_channel,
            kernel_size=3,
            padding=1,
            groups=hidden_channel,
        )
        self.act2 = nn.GELU()
        self.norm2 = build_norm_layer(norm_cfg, hidden_channel)

        self.conv3 = nn.Conv2d(hidden_channel, num_channel, kernel_size=1)
        self.act3 = nn.GELU()
        self.norm3 = build_norm_layer(norm_cfg, num_channel)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, "The shape doesn't match H, W."
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.DWConv(x)))
        x = self.act3(self.norm3(self.conv3(x)))

        # reshape BCHW to BNC
        # x = x.flatten(2).transpose(1, 2).contiguous()
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()
        # print("after DW-CNN shape : ", x.shape)
        return x


class W_MSA(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        window_size,
        qkv_bias=True,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        with_rpe=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5
        self.with_rpe = with_rpe
        if self.with_rpe:
            self.learnable_rpe_table = nn.Parameter(
                torch.randn((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            self.rpe_index = self.get_relative_distances(window_size)

        # qkv를 합쳐서 레이어 생성
        self.qkv = nn.Linear(self.embed_dims, self.embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_relative_distances(window_size):
        Wh, Ww = window_size
        indices = torch.tensor(np.array([[x, y] for x in range(Ww) for y in range(Wh)]))
        distances = indices[None, :, :] - indices[:, None, :]

        x_table = distances[:, :, 0] + Ww - 1
        y_table = distances[:, :, 1] + Wh - 1

        x_table *= 2 * Ww - 1
        relative_position_coords = x_table + y_table
        relative_position_coords = relative_position_coords.transpose(0, 1).contiguous()
        return relative_position_coords

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        # B : batch * window_num
        # N : num_feature per patch
        # C : dim per feature
        B, N, C = x.shape
        # print("B N C shape : ", B,N,C , self.head_dims)

        # [qkv, B, num_heads, N , dim ]
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print("QKV shape :", q.shape)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.with_rpe:
            relative_position_bias = self.learnable_rpe_table[self.rpe_index]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

            attn += relative_position_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LW_MSA(nn.Module):
    def __init__(
        self,
        num_channel,
        num_heads,
        window_size,
        qkv_bias=True,
        drop_rate=0.0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="SyncBN"),
        transformer_norm_cfg=dict(type="LN", eps=1e-6),
    ):
        super().__init__()
        norm_cfg = copy.deepcopy(norm_cfg)
        transformer_norm_cfg = copy.deepcopy(transformer_norm_cfg)

        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.transformer_norm_cfg = transformer_norm_cfg
        self.num_heads = num_heads
        self.num_channel = num_channel

        self.attn = W_MSA(
            embed_dims=num_channel,
            num_heads=num_heads,
            window_size=self.window_size,
            qkv_bias=qkv_bias,
            attn_drop_rate=drop_rate,
            proj_drop_rate=drop_rate,
            with_rpe=True,
        )

    def forward(self, x, H, W):
        # print("LW-MSA shape :",x.shape)
        B, N, C = x.shape
        # Normalized 돤 input을 다시 (B, H, W, C)로 shape 바꿔주고,
        x = x.view(B, H, W, C)

        Wh, Ww = self.window_size
        # Local window 사용을 위해 기존 x에 padding 넣어줘여함.
        pad_H = math.ceil(H / Wh) * Wh - H
        pad_W = math.ceil(W / Ww) * Ww - W

        x = pad(x, (0, 0, pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2))
        # Local window size에 맞게 reshape 해줌
        # (B, num_patch_h, window_size_h, num_patch_w, window_size_w, C)
        x = x.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C)
        # (B, num_patch_h, num_patch_w, window_h, window_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        B, P_h, P_w, Wh, Ww, C = x.shape
        x = x.reshape(B * P_h * P_w, Wh * Ww, C)
        x = self.attn(x)

        # reverse permutation
        x = x.reshape(B, P_h, P_w, Wh, Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, P_h * Wh, P_w * Ww, C)

        # de-pad
        x = x[:, pad_H // 2 : H + pad_H // 2, pad_W // 2 : W + pad_W // 2]

        # print("after MSA shape : ", x.shape)
        return x.reshape(B, N, C)


class HRFormer(nn.Module):
    blocks_dict = {"BOTTLENECK": Bottleneck_block, "HRFORMERBLOCK": HRFormer_block}

    def __init__(
        self,
        extra,
        in_channels=3,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
    ):
        norm_cfg = copy.deepcopy(norm_cfg)
        super(HRFormer, self).__init__()

        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.transformer_norm = copy.deepcopy(dict(type="LN", eps=1e-6))

        depths = [
            extra[stage]["num_blocks"][0] * extra[stage]["num_modules"]
            for stage in ["stage2", "stage3", "stage4"]
        ]
        depth_s2, depth_s3, _ = depths

        drop_path_rate = extra["drop_path_rate"]

        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, sum(depths))]

        extra["stage2"]["drop_path_rates"] = dpr[0:depth_s2]
        extra["stage3"]["drop_path_rates"] = dpr[depth_s2 : depth_s2 + depth_s3]
        extra["stage4"]["drop_path_rates"] = dpr[depth_s2 + depth_s3 :]

        self.extra = extra
        self.stage1_cfg = extra["stage1"]
        self.stage2_cfg = extra["stage2"]
        self.stage3_cfg = extra["stage3"]
        self.stage4_cfg = extra["stage4"]

        # self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        # self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = nn.Conv2d(
            in_channels,
            self.stage1_cfg["in_channel"],
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=False,
        )
        self.norm1 = build_norm_layer(self.norm_cfg, self.stage1_cfg["in_channel"])
        self.conv2 = nn.Conv2d(
            self.stage1_cfg["in_channel"],
            self.stage1_cfg["in_channel"],
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=False,
        )
        self.norm2 = build_norm_layer(self.norm_cfg, self.stage1_cfg["in_channel"])
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_stage_1(
            block=self.blocks_dict[self.stage1_cfg["block"]],
            in_channels=self.stage1_cfg["in_channel"],
            out_channels=self.stage1_cfg["out_channel"],
            num_blocks=self.stage1_cfg["num_blocks"],
        )
        self.transition1 = self._make_transition_layer(
            self._last_channel, self.stage2_cfg["num_channels"]
        )

        self.stage2 = self._make_stage_HRFormer(self.stage2_cfg, self.transition1)

        self.transition2 = self._make_transition_layer(
            self._last_channel, self.stage3_cfg["num_channels"]
        )

        self.stage3 = self._make_stage_HRFormer(self.stage3_cfg, self.transition2)

        self.transition3 = self._make_transition_layer(
            self._last_channel, self.stage4_cfg["num_channels"]
        )

        self.stage4 = self._make_stage_HRFormer(self.stage4_cfg, self.transition3)

    def _make_transition_layer(self, last_branch_channel, channel):
        last_branch_num = len(last_branch_channel)
        cur_branch_num = len(channel)

        transition_layers = []
        for i in range(cur_branch_num):
            if i < last_branch_num:
                if last_branch_channel[i] != channel[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=last_branch_channel[i],
                                out_channels=channel[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, channel[i]),
                            self.relu,
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                transition_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=last_branch_channel[-1],
                            out_channels=channel[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        build_norm_layer(self.norm_cfg, channel[i]),
                        self.relu,
                    )
                )
        return nn.ModuleList(transition_layers)

    def _make_stage_1(self, block, in_channels, out_channels, num_blocks, stride=1):
        """make layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, out_channels),
            )

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expansion=4,
                downsample=downsample,
                norm_cfg=self.stage1_cfg["norm_cfg"],
            )
        )

        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=4,
                    norm_cfg=self.stage1_cfg["norm_cfg"],
                )
            )

        # print(layers)
        self._last_channel = [
            out_channels,
        ]
        return nn.Sequential(*layers)

    def _make_stage_HRFormer(self, cfg: dict, branches):
        window_size = cfg["window_sizes"]
        num_modules = cfg["num_modules"]
        num_channel = cfg["num_channels"]
        num_branch = cfg["num_branch"]
        num_blocks = cfg["num_blocks"]
        num_heads = cfg["num_heads"]
        mlp_ratios = cfg["mlp_ratios"]
        drop_path = cfg["drop_path_rates"]
        with_rpe = True

        module_list = []

        for _ in range(num_modules):
            module_list.append(
                HRFormer_module(
                    window_size,
                    num_channel,
                    num_branch,
                    num_blocks,
                    num_heads,
                    mlp_ratios,
                    drop_path[num_blocks[0] * _ : num_blocks[0] * (_ + 1)],
                    with_rpe,
                    self.norm_cfg,
                    self.transformer_norm,
                )
            )
        self._last_channel = num_channel
        return nn.Sequential(*module_list)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.checkpoint = self._load_from_local(
                pretrained,
            )
            self.load_state_dict(self.checkpoint["state_dict"])
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

    def _load_from_local(self, filename, map_location=None):
        if not osp.isfile(filename):
            raise IOError(f"{filename} is not a checkpoint file")
        checkpoint = torch.load(filename, map_location=map_location)

        changed_checkpoint = {}
        changed_model = OrderedDict()
        filtered_layer_name = ["cls_token", "keypoint_head"]
        backbone_name = "backbone"
        for state_dict_name, model in checkpoint.items():
            for layer, params in model.items():
                layer_without_backbone_name = layer
                if backbone_name in layer:
                    layer_without_backbone_name = layer[len(backbone_name) + 1 :]
                if any(s in layer_without_backbone_name for s in filtered_layer_name):
                    continue
                changed_model[layer_without_backbone_name] = params
            changed_checkpoint[state_dict_name] = changed_model

        return changed_checkpoint

    def weight_init_hardcode(filename):
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        # stage 1 끝
        x = self.stage1(x)

        # make branch
        x_list = []
        for i in range(len(self.transition1)):
            if self.transition1[i] != None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # stage 2 끝
        y_list = self.stage2(x_list)

        # transition 2 계산
        x_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] != None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # stage 3 끝
        y_list = self.stage3(x_list)

        # transition 3 계산
        x_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] != None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # stage 3 끝
        y_list = self.stage4(x_list)

        return y_list[0]
