
_base_ = [
    './upernet_transnext.py',
    'mmsegext::_base_/datasets/ade20k_512_tta_without_ratio.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_160k.py'
]
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=None,
    paramwise_cfg=dict(
        custom_keys=dict(
            query_embedding=dict(decay_mult=0.0),
            relative_pos_bias_local=dict(decay_mult=0.0),
            cpb=dict(decay_mult=0.0),
            temperature=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
model = dict(
    backbone=dict(
        pretrained=None,
        type='ext-TransNeXt-tiny',
        pretrain_size=224,
        img_size=800,
        is_extrapolation=False),
    decode_head=dict(in_channels=[72, 144, 288, 576], num_classes=150),
    auxiliary_head=dict(in_channels=288, num_classes=150))
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