norm_cfg = dict(type="BN", requires_grad=True)
extra = dict(
    drop_path_rate=0.1,
    with_rpe=True,
    stage1=dict(
        block="BOTTLENECK",
        in_channel=64,
        out_channel=256,
        expantion=4,  # mlp_ratio
        stride=1,
        diliation=1,
        num_blocks=2,  # int
        norm_cfg=norm_cfg,
    ),
    stage2=dict(
        num_modules=1,
        num_branch=2,
        block="HRFormerBlock",
        num_blocks=(2, 2),
        num_channels=(32, 64),
        num_heads=[1, 2],
        mlp_ratios=[4, 4],
        window_sizes=[7, 7],
    ),
    stage3=dict(
        num_modules=4,
        num_branch=3,
        block="HRFormerBlock",
        num_blocks=(2, 2, 2),
        num_channels=(32, 64, 128),
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        window_sizes=[7, 7, 7],
    ),
    stage4=dict(
        num_modules=2,
        num_branch=4,
        block="HRFormerBlock",
        num_blocks=(2, 2, 2, 2),
        num_channels=(32, 64, 128, 256),
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        window_sizes=[7, 7, 7, 7],
    ),
    joint_num=17,
    uncertainty_head=dict(
        in_channel=32,
        out_channel=34,
    ),
    uncertainty_channel_head=dict(
        in_channel=32,
        out_channel=2,
        num_kp=17,
    ),
)
