_base_ = [
    './_base_/faster_rcnn_r50_fpn_syncbn.py',
    './_base_/coco_detection.py',
    './_base_/schedule_20e.py',
    './_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['mmdet.models.rand_box',
             'mmdet.models.contrastive_head'],
    allow_failed_imports=False)
