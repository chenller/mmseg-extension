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
        type='ext-FlashInternImage',
        channels=80,
        core_op='DCNv4',
        depths=[4, 4, 21, 4, ],
        drop_path_rate=0.3,
        dw_kernel_size=3,
        groups=[5, 10, 20, 40, ],
        # init_cfg=dict(
        #     checkpoint=
        #     'https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_s_1k_224.pth',
        #     type='Pretrained'),
        layer_scale=1.0,
        mlp_ratio=4.0,
        norm_layer='LN',
        offset_scale=1.0,
        out_indices=(0, 1, 2, 3,),
        post_norm=True,
        with_cp=True),
    decode_head=dict(
        type='UPerHead',
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[80, 160, 320, 640, ],
        in_index=[0, 1, 2, 3, ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=150,
        pool_scales=(1, 2, 3, 6,),
    ),

    auxiliary_head=dict(
        type='FCNHead',
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=320,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=150,
        num_convs=1,
    ),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
)

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
