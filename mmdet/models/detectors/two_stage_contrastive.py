import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import (DETECTORS, build_backbone, build_contrastive_head, build_head, build_neck,
                       build_rand_box)
from .base import BaseDetector


@DETECTORS.register_module()
class TwoStageDetectorForContrastive(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 rand_box,
                 contrastive_head,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetectorForContrastive, self).__init__()
        self.backbone = build_backbone(backbone)

        self.gen_rand_box = build_rand_box(rand_box)
        self.contrastive_cfg = train_cfg.pop("contrastive", None)
        assert self.contrastive_cfg is not None, \
            "you need to add contrastive_cfg to train_cfg"

        self.contrastive_head = build_contrastive_head(contrastive_head)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if self.contrastive_cfg['train_rpn'] and roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetectorForContrastive, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        device = img.device
        num_img = img.size(0)

        # prepare different views
        with torch.no_grad():
            img_1 = img[:, :3, :, :].clone()
            img_2 = img[:, 3:, :, :].clone()

        pseude_gt_bboxes = self.gen_rand_box(num_img, device, img_metas)

        x_1 = self.extract_feat(img_1)
        x_2 = self.extract_feat(img_2)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_losses_1 = self.rpn_head.forward_train(
                x_1,
                img_metas,
                pseude_gt_bboxes)
            rpn_losses_1 = self.identify(rpn_losses_1, 1)
            losses.update(rpn_losses_1)
            rpn_losses_2 = self.rpn_head.forward_train(
                x_2,
                img_metas,
                pseude_gt_bboxes)
            rpn_losses_2 = self.identify(rpn_losses_2, 2)
            losses.update(rpn_losses_2)

        proposal_list = pseude_gt_bboxes

        roi_feats_1 = self.roi_head.forward_train(x_1, proposal_list)
        roi_feats_2 = self.roi_head.forward_train(x_2, proposal_list)

        # forward projection, prediction and loss
        constrastive_losses = self.contrastive_head(roi_feats_1,
                                                    roi_feats_2)
        losses.update(constrastive_losses)
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        pass

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        pass

    def aug_test(self, imgs, img_metas, rescale=False):
        pass

    def identify(self, loss, id):
        assert isinstance(loss, dict), "loss is not dict()."
        assert isinstance(id, int), "id is not int."
        out = dict()
        for k, v in loss.items():
            out[f"{k}_{id}"] = v
        return out
