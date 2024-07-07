_base_ = [
    'mmseg::_base_/datasets/ade20k.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
num_classes = 150
data_preprocessor = dict(
    type='SegDataPreProcessor', _scope_='mmseg',
    mean=[123.675, 116.28, 103.53, ],
    std=[58.395, 57.12, 57.375, ],
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
model = dict(
    type='EncoderDecoder', _scope_='mmseg',
    data_preprocessor=data_preprocessor,
    pretrained='./convnextv2_base_22k_384_ema.pt',
    backbone=dict(
        type='ext-ConvNeXtV2',
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    decode_head=dict(
        type='UPerHead',
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[128, 256, 512, 1024, ],
        in_index=[0, 1, 2, 3, ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        pool_scales=(1, 2, 3, 6,),
        loss_decode=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=512,
        in_index=2,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        num_convs=1,
        loss_decode=dict(loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512,), stride=(341, 341,)),
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'bias': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)}
    )
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
