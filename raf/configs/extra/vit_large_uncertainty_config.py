norm_cfg = dict(type="BN", requires_grad=True)
extra = dict(
    drop_path_rate=0.5,
    joint_num=17,
    backbone=dict(
        type="ViT",
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.5,
        lr_decay_rate=0.8,
    ),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=1024,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            final_conv_kernel=1,
        ),
        out_channels=17,
    ),
    uncertainty_head=dict(
        in_channel=1024,
        hid_channel=64,
        out_channel=34,
    ),
    uncertainty_channel_head=dict(
        in_channel=1024,
        out_channel=1,
        num_kp=17,
    ),
)
