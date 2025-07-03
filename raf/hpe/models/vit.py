import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import os.path as osp

from collections import OrderedDict
from functools import partial
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, in_channels=3, embed_dim=768) -> None:
        super().__init__()
        img_size = to_2tuple(img_size) # tuple로 만들어주는 함수
        patch_size = to_2tuple(patch_size)
        self.n_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) # input image에서 나올 수 있는 patch의 개수 # 192
        self.patch_shape = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size[0],
            padding=2,
        )

    def forward(self, x):
        # print(f"patch_size => {self.patch_shape}")
        x = self.proj(x)
        # print(f"proj 후 x => {x.shape}")
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).permute(0, 2, 1)
        return x, (Hp, Wp)

class GlobalPosEmbed(nn.Module):
    def __init__(self, embed_dim, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.embed_dim = embed_dim // 2 # sin, cosin의 주기적인 성질을 각각 반씩 이용하기 위해서
        self.normalize = normalize
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.embed_layer = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding= 1, groups = embed_dim)
        
    def forward(self, x):
        b, n, c = x.shape # (batch_size, patch_h_n * patch_w_n + 1, self.embed_dim)
        # print(f"batch => {b}, n => {n}, c => {c}")

        # ResFormer의 코드를 4:3 비율을 가지고 있는 ViTPose에 맞게 변경한 코드
        unit_scale = int(((n-1) // 12) ** 0.5)
        patch_h_n = 4 * unit_scale
        patch_w_n = 3 * unit_scale
        not_mask = torch.ones((b, patch_h_n, patch_w_n), device = x.device)

        y_embed = not_mask.cumsum(1, dtype=torch.float32) # height에 대해서 누적으로 점차 더해짐.
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # width에 대해서 누적으로 점차 더해짐.
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale # y_embed의 마지막 height 값으로(모든 값들이 누적된 값) 나눠주고 scaling 함으로써 정규화함. (모든 값들이 0 ~ scale)
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale # x_embed의 마지막 width 값으로(모든 값들이 누적된 값) 나눠주고 scaling 함으로써 정규화함. (모든 값들이 0 ~ scale)
        dim_t = torch.arange(self.embed_dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embed_dim) # 2 * (dim_t // 2) => [0, 0, 2, 2, ... ] # og transformer의 positional encoding 식.
        pos_x = x_embed[:, :, :, None] / dim_t # shape => (batch_size, patch_h_n, patch_w_n, self.embed_dim)
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # sin과 cos값을 더해주기 위해서 새로운 차원 추가, 더한 후 다시 통합.
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # B, C, H, W # 이 부분을 통해서 원래의 embed_dim 복원 permute 하기 전후 변화 => (batch_size, patch_h_n, patch_w_n, self.embed_dim) => (batch_size, self.embed_dim, patch_h_n, patch_w_n)
        pos = self.embed_layer(pos).reshape(b, c, -1).transpose(1, 2) # (batch_size, self.embed_dim, patch_h_n, patch_w_n) => (batch_size, self.embed_dim, patch_h_n * patch_w_n) => (batch_size, patch_h_n * patch_w_n, self.embed_dim)
        pos_cls = torch.zeros((b, 1, c), device = x.device) # (batch_size, 1, self.embed_dim)
        pos =  torch.cat((pos_cls, pos),dim=1) # (batch_size, patch_h_n * patch_w_n + 1, self.embed_dim)
        return pos + x

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

        unit_scale = int(((N-1) // 12) ** 0.5)
        H = 4 * unit_scale # 현재 feature map의 height
        W = 3 * unit_scale # 현재 feature map의 width

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


class ViT(nn.Module):
    def __init__(
        self,
        img_size=(256, 192),
        patch_size=16,
        in_channels=3,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        pos_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        freeze_attn=False,
        freeze_ffn=False,
        frozen_stages=-1,
        use_gpe=False,
        use_lpe=False,
        use_gap=False,
    ):
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)) # (1, 16*12 + 1, 768)
        
        self.use_gpe = use_gpe
        self.use_gap = use_gap
        if self.use_gpe:
            print("self.use_gpe 쓰고 있음!!!! ------------------------------------------")
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # (1, 1, embed_dim)
            self.global_pos_embed = GlobalPosEmbed(embed_dim)
            self.pos_drop = nn.Dropout(p=pos_drop_rate)
        else:
            print("self.use_gpe 안 쓰고 있음!!!! ------------------------------------------")
            self.cls_token = None
            self.global_pos_embed = None
            self.pos_drop = None

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
                    drop_path=drop_path_ratio[i],
                    norm_layer=norm_layer,
                    use_lpe=use_lpe
                )
                for i in range(depth)
            ]
        )

        self.last_norm = norm_layer(embed_dim)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

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
        B, C, H, W = x.shape # torch.Size([B, 3, 256, 192])
        x, (Hp, Wp) = self.patch_embed(x) # torch.Size([B, 16*12, 1280])
        
        if self.use_gpe:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks # torch.Size([B, 1, embed_dim])
            x = torch.cat((cls_token, x), dim=1) # torch.Size([B, 16*12 + 1, embed_dim])
            x = self.pos_drop(self.global_pos_embed(x)) # torch.Size([B, 16*12 + 1, embed_dim])
        else:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1] # pos_embed: (1, 16*12 + 1, 1280)
            # x.shape = (B, 16*12, 1280) = (B, 16*12, embed_dim)

        # transformer encoder
        cnt = 0
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        x = self.last_norm(x)
        
        if self.use_gpe:
            global_token = x[:, :1]
            x = x[:, 1:] + global_token
        
        xp = rearrange(x, "b (h w) d-> b d h w", h=Hp).contiguous()
        return xp