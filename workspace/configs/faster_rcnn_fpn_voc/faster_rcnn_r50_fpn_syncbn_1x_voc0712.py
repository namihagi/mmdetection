_base_ = [
    '../../../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/voc0712.py',
    '../../../configs/_base_/default_runtime.py'
]
# for model
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False),
    neck=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    roi_head=dict(
        bbox_head=dict(num_classes=20)))
# for dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        times=1,
        dataset=dict(
            pipeline=train_pipeline)),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    step=[11, 15])
# runtime settings
total_epochs = 17
