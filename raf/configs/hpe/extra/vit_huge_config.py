norm_cfg = dict(type="BN", requires_grad=True)
extra = dict(
    drop_path_rate=0.5,
    joint_num=17,
    backbone=dict(
        type="ViT",
        img_size=(384, 288),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        lr_decay_rate=0.85,
    ),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=1280,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            final_conv_kernel=1,
        ),
        out_channels=17,
    ),
    # uncertainty_head와 uncertainty_channel_head 제거됨 (사용하지 않음)
) 