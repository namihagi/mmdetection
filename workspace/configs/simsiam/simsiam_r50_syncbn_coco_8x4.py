_base_ = [
    './_base_/simsiam_r50_syncbn.py',
    './_base_/coco_detection.py',
    './_base_/schedule_100e_annealing.py',
    './_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['mmdet.models.rand_box',
             'mmdet.models.contrastive_head'],
    allow_failed_imports=False)
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type="SimsiamAugWithCrop"),
    dict(type='NormalizeForContrastive', **img_norm_cfg),
    dict(type='DefaultFormatBundleForContrastive'),
    dict(
        type='Collect', keys=['img'],
        meta_keys=('filename', 'ori_filename', 'ori_shape',
                   'img_shape', 'pad_shape', 'scale_factor',
                   'img_norm_cfg')),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
