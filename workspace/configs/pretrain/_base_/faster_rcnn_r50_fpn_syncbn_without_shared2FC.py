model = dict(
    type='FasterRCNNForContrastive',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    rand_box=dict(
        type="RandBox",
        flip=False,
        nms_thr=0.7,
        num_of_init_boxes=2000,
        min_scale_rate=0.01,
        min_num_of_final_box=5,
        max_num_of_final_box=50),
    contrastive_head=dict(
        type='ContrastiveHead',
        each_view_loss_weight=0.5,
        projection=dict(
            type='ProjectionMLP',
            in_channels=256,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False),
        prediction=dict(
            type='PredictionMLP',
            in_channels=256,
            hidden_channels=128,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False)),
    rpn_head=dict(
        type='RPNHeadForContrastive',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.5)),
    roi_head=dict(
        # type='StandardRoIHead',
        type='StandardRoIHeadForContrasitve',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='BBoxHeadForContrastive',
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=256)),
    # model training and testing settings
    train_cfg=dict(
        contrastive=dict(
            train_rpn=True),
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
