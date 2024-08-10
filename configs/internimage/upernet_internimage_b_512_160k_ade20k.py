_base_ = [
    'mmsegext::_base_/datasets/ade20k_512_tta_without_ratio.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_160k.py'
]
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=(512, 512),
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ext-InternImage',
        core_op='DCNv3',
        channels=112,
        depths=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        mlp_ratio=4.0,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='/path/to/internimage_b_1k_224.pth',
        #     # 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth',
        # )
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[112, 224, 448, 896],
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
        in_channels=448,
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

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    constructor='ext-LayerDecayOptimizerConstructor-InternImage',
    paramwise_cfg=dict(num_layers=33, layer_decay_rate=1.0, depths=[4, 4, 21, 4]),
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
