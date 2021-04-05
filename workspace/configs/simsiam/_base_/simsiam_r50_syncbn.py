# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SimSiam',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='caffe'),
    contrastive_head=dict(
        type='ContrastiveHead',
        each_view_loss_weight=0.5,
        projection=dict(
            type='ProjectionMLP',
            in_channels=2048,
            norm_cfg=norm_cfg,
            norm_eval=False),
        prediction=dict(
            type='PredictionMLP',
            in_channels=2048,
            norm_cfg=norm_cfg,
            norm_eval=False)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())
