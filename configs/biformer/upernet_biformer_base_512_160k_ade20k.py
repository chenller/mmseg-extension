# This configuration comes from the website 'https://github.com/rayleizhu/BiFormer/blob/public_release/semantic_segmentation/configs/ade20k/upernet.biformer_small.py',
# and we rename the file name to 'upernet_biformer_base_512_160k_ade20k.py'
_base_ = [
    'mmsegext::_base_/datasets/ade20k_512_tta_without_ratio.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_160k.py'
]
# backbone pretrained weight 'biformer_base_in1k.pth'
# https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHI_XPhoadjaNxtO/root/content
pretrained = './pretrained/biformer/biformer_base_best.pth'

data_preprocessor = dict(
    type='SegDataPreProcessor', _scope_='mmseg',
    size=(512, 512),
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=pretrained,
    backbone=dict(
        type='ext-BiFormer',
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[3, 3, 3, 3],
        n_win=8,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        use_checkpoint_stages=[],
        drop_path_rate=0.4),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01, )
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper', optimizer=optimizer, clip_grad=None,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)}
    )
)
# learning policy
param_scheduler = [
    # 线性学习率预热调度器
    dict(type='LinearLR',
         start_factor=1e-6,
         by_epoch=False,  # 按迭代更新学习率
         begin=0,
         end=1500),  # 预热前 50 次迭代
    # 主学习率调度器
    dict(
        type='PolyLR',
        eta_min=0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False)
]
