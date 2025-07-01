import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import os.path as osp

from collections import OrderedDict
from functools import partial
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import numpy as np
import math

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_lpe=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        all_head_dim = self.head_dim * num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_lpe = use_lpe

        # LPE를 위한 추가
        # ------------------------------------------------------------------------
        if self.use_lpe:
            self.get_v = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, stride=1, padding=1,groups=self.head_dim)
            nn.init.zeros_(self.get_v.weight)
            nn.init.zeros_(self.get_v.bias)
        # ------------------------------------------------------------------------
    
    def get_local_pos_embed(self, x):
        B, _, N, C = x.shape
        # H = W = int(np.sqrt(N-1)) # 이거는 이미지 크기가 정방형이기 때문에 이렇게 한 것.

        unit_scale = int(((N-1) // 12) ** 0.5)
        H = 4 * unit_scale
        W = 3 * unit_scale

        x = x[:, :, 1:].transpose(-2, -1).contiguous().reshape(B * self.num_heads, -1, H, W)
        local_pe = self.get_v(x).reshape(B, -1, C, N-1).transpose(-2, -1).contiguous() # B, H, N-1, C
        local_pe = torch.cat((torch.zeros((B, self.num_heads, 1, C), device=x.device), local_pe), dim=2) # B, H, N, C
        return local_pe

    def forward(self, x):
        # qkv shape: (batch_size, number of patches, 3 * head_num * head_dim)
        # torch.Size([16, 432, 1280]) / global pos embed 적용했을 때는 torch.Size([16, 432 + 1, 1280])
        qkv = self.qkv(x)

        # qkv shape: (3, batch_size, head_num, number of patches, head_dim)
        qkv = rearrange(qkv, "b n (c h d) -> c b h n d", h=self.num_heads, c=3)
        # print(f"rearrange_qkv의 shape => {qkv.shape}")

        # q, k, v shape: (batch_size, head_num, number of patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention_score = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        attention_score = attention_score * self.scale

        # attention shape: (batch_size, head_num, number of patches, number of patches)
        attention_map = attention_score.softmax(dim=-1)
        attention_map = self.attn_drop(attention_map)

        x = torch.einsum("bham, bhmv -> bhav", attention_map, v)
        # ------------------------------------------------------------------------
        if self.use_lpe:
            local_pe = self.get_local_pos_embed(v) # 이게 추가됨.
            x = x + local_pe
        # ------------------------------------------------------------------------

        # output shape: (batch_size, number of patches, embedding_dim)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = in_features or out_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


class Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        use_lpe=False
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, qkv_bias, attn_drop, drop, use_lpe=use_lpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_features=mlp_hidden_dim, drop=drop)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT_server(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        freeze_attn=False,
        freeze_ffn=False,
        frozen_stages=-1,
        use_gpe=False,
        use_lpe=False,
    ):
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.use_gpe = use_gpe

        self.use_checkpoint = use_checkpoint
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.frozen_stages = 11

        drop_path_ratio = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    drop=drop_rate,
                    drop_path=drop_path_ratio[i+1],
                    norm_layer=norm_layer,
                    use_lpe=use_lpe
                )
                for i in range(depth-1)
            ]
        )

        self.last_norm = norm_layer(embed_dim)

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

    def forward(self, x):
        Hp = int(np.sqrt((x.shape[1] - 1) / 12) * 4)
        # transformer encoder
        for block in self.blocks:
            x = block(x)

        x = self.last_norm(x)
        
        kd_output = x[:, 0]
        
        # ----------------------------------------------------------------------
        if self.use_gpe:
            global_token = x[:, :1]
            x = x[:, 1:] + global_token
        # ----------------------------------------------------------------------
        
        xp = rearrange(x, "b (h w) d-> b d h w", h=Hp).contiguous()
        return xp, kd_output
