#!/bin/bash

cd /mmdetection
bash ./tools/dist_train.sh \
    /mmdetection/workspace/configs/pretrain/faster_rcnn_r50_pytorch_c4_20e_coco.py \
    8 \
    --work-dir /mmdetection/workspace/work_dirs/faster_rcnn_pytorch_c4_voc_1/pretrain \
    --no-validate
