_base_ = [
    './_base_/faster_rcnn_r50_syncbn_c4.py',
    './_base_/coco_detection.py',
    './_base_/schedule_20e.py',
    './_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['mmdet.models.rand_box',
             'mmdet.models.contrastive_head'],
    allow_failed_imports=False)
model = dict(
    pretrained="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar")
# use pytorch img_norm
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="SimsiamAugmentation",
        to_rgb=True,
        jitter_param=dict(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.4),
        jitter_p=0.8,
        grayscale_p=0.2,
        gaussian_sigma=[0.1, 2.0],
        gaussian_p=0.5),
    dict(
        type='ResizeForContrastive',
        img_scale=[(1333, 800), (1333, 640)],
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='NormalizeForContrastive', **img_norm_cfg),
    dict(type='PadForContrastive', size_divisor=32),
    dict(type='DefaultFormatBundleForContrastive'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
