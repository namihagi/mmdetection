_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False))
data = dict(
    train=dict(
        ann_file='../data/coco/annotations/instances_train2017.json',
        img_prefix='../data/coco/train2017/'),
    val=dict(
        ann_file='../data/coco/annotations/instances_val2017.json',
        img_prefix='../data/coco/val2017/'),
    test=dict(
        ann_file='../data/coco/annotations/instances_val2017.json',
        img_prefix='../data/coco/val2017/'))
