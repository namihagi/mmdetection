_base_ = [
    './_base_/faster_rcnn_r50_fpn.py',
    './_base_/coco_detection.py',
    './_base_/schedule_1x.py',
    './_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['mmdet.models.rand_box',
             'mmdet.models.contrastive_head'],
    allow_failed_imports=False)
model = dict(
    same_scale=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="SimsiamAugmentation",
        to_rgb=True,
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
    train=dict(pipeline=train_pipeline))
