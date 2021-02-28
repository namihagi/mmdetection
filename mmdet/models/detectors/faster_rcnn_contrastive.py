from ..builder import DETECTORS
from .two_stage_contrastive import TwoStageDetectorForContrastive


@DETECTORS.register_module()
class FasterRCNNForContrastive(TwoStageDetectorForContrastive):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rand_box,
                 contrastive_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 rpn_head=None,
                 pretrained=None,
                 same_scale=True):
        super(FasterRCNNForContrastive, self).__init__(
            backbone=backbone,
            rand_box=rand_box,
            contrastive_head=contrastive_head,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            same_scale=same_scale)
