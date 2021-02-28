_base_ = [
    '../../../configs/_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../../../configs/_base_/datasets/voc0712.py',
    '../../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False),
    roi_head=dict(
        shared_head=dict(
            norm_cfg=norm_cfg,
            norm_eval=False),
        bbox_head=dict(num_classes=20)))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800), (1333, 480)],
        keep_ratio=True),
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
        img_scale=(1333, 800),
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
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[11, 15])
total_epochs = 17
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
