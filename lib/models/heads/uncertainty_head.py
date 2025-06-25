import torch.nn as nn

from lib.utils.utils import build_norm_layer
import copy

class UncertaintySimpleHead(nn.Module):
    def __init__(self, config, extra=None) -> None:
        super().__init__()
        in_channel = 384
        hid_channel = 64
        out_channel = 34
        if extra:
            in_channel = extra["uncertainty_head"]["in_channel"]
            hid_channel = extra["uncertainty_head"]["hid_channel"]
            out_channel = extra["uncertainty_head"]["out_channel"]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channel,
                hid_channel,
                kernel_size=3,
            ),
            nn.BatchNorm2d(hid_channel),
            nn.GELU(),
            nn.Conv2d(
                hid_channel,
                out_channel,
                kernel_size=3,
            ),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )

    def avg_pool(self, x):
        return x.mean((2, 3))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        sigma = x.view(b, -1, 2)
        return sigma

class UncertaintyDepthWiseSeparableHead(nn.Module):
    def __init__(self,
                extra,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
            ):
        super().__init__()

        in_channel = extra['in_channel']
        out_channel = extra['out_channel']

        norm_cfg = copy.deepcopy(norm_cfg)

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.act1 = nn.GELU()
        self.norm1 = build_norm_layer(norm_cfg, out_channel)
        
        self.DWConv = nn.Conv2d(out_channel, out_channel, kernel_size=3, 
                             padding=1, groups= out_channel)
        self.act2 = nn.GELU()
        self.norm2 = build_norm_layer(norm_cfg, out_channel)
        
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.act3 = nn.GELU()
        self.norm3 = build_norm_layer(norm_cfg, out_channel)

        
    def avg_pool(self, x):
        return x.mean((2,3))
    
    def forward(self,x):
        # for HRFormer
        # x : [batch, 32, 64, 48]
        b, c, h, w = x.shape
        
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.DWConv(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        
        # x : [batch,34,64,48]
        x = self.avg_pool(x)
        
        # x : [batch,34,]
        x = x.view(b,-1,2)
        return x
        

# For ViT
class UncertaintyDeconvDepthWiseChannelHead(nn.Module):
    def __init__(
        self,
        extra,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
    ):
        super().__init__()

        in_channel = extra["in_channel"]
        self.in_channels = in_channel
        out_channel = extra["out_channel"]
        num_kp = extra["num_kp"]

        norm_cfg = copy.deepcopy(norm_cfg)

        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        self.final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.conv1 = nn.Conv2d(32, num_kp, kernel_size=1)
        self.act1 = nn.GELU()
        self.norm1 = build_norm_layer(norm_cfg, num_kp)

        self.DWConv1 = nn.Conv2d(num_kp, num_kp, kernel_size=3, padding=1, groups=num_kp)
        self.act2 = nn.GELU()
        self.norm2 = build_norm_layer(norm_cfg, num_kp)

        self.conv3 = nn.Conv2d(num_kp, out_channel, kernel_size=1)
        self.act3 = nn.GELU()
        self.norm3 = build_norm_layer(norm_cfg, out_channel)

        self.DWConv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel
        )
        self.act4 = nn.GELU()
        self.norm4 = build_norm_layer(norm_cfg, out_channel)

        self.conv5 = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.act5 = nn.GELU()
        self.norm5 = build_norm_layer(norm_cfg, out_channel)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        kernel = 4
        padding = 1
        output_padding = 0
        for i in range(num_layers):
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def avg_pool(self, x):
        return x.mean((2, 3))

    def forward(self, x):
        # for ViT
        # x : [batch, 384, 16, 12]
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        # x : [batch, 32, 16, 12]
        x = self.act1(self.norm1(self.conv1(x)))
        # print(x.shape)
        x = self.act2(self.norm2(self.DWConv1(x)))
        # print(x.shape)
        # x : [batch,17,64,48]
        x = self.act3(self.norm3(self.conv3(x)))
        x = self.act4(self.norm4(self.DWConv2(x)))
        x = self.act5(self.norm5(self.conv5(x)))
        # print(x.shape)
        # x : [batch,2,64,48]
        return x
