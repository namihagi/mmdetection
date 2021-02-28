_base_ = [
    './_base_/faster_rcnn_r50_syncbn_caffe_c4.py',
    './_base_/coco_detection.py',
    './_base_/schedule_20e.py',
    './_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['mmdet.models.rand_box',
             'mmdet.models.contrastive_head'],
    allow_failed_imports=False)
model = dict(
    same_scale=False,
    rand_box=dict(
        flip=True))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="SimsiamAugmentation",
        to_rgb=False,
        same_scale=False,
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
        same_scale=False,
        img_scale=[(1333, 800), (640, 480)],
        keep_ratio=False),
    dict(
        type='RandomFlipForContrastive',
        same_scale=False,
        flip_ratio=0.9),
    dict(
        type='NormalizeForContrastive',
        same_scale=False,
        **img_norm_cfg),
    dict(
        type='PadForContrastive',
        same_scale=False,
        size_divisor=32),
    dict(
        type='DefaultFormatBundleForContrastive',
        same_scale=False),
    dict(type='CollectForContrastive',
         same_scale=False,
         keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001,
    momentum=0.9, weight_decay=0.0001)