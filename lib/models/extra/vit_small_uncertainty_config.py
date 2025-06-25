norm_cfg = dict(type="BN", requires_grad=True)
extra = dict(
    drop_path_rate=0.1,
    joint_num=17,
    backbone=dict(
        type="ViT",
        img_size=(256, 192),
        # img_size=(192, 144),
        # img_size=(128, 96),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        lr_decay_rate=0.75,
    ),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=384,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            final_conv_kernel=1,
        ),
        out_channels=17,
    ),
    associate_keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=384,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            final_conv_kernel=1,
        ),
        out_channels=16,
    ),
    uncertainty_head=dict(
        in_channel=384,
        hid_channel=64,
        out_channel=34,
    ),
)
