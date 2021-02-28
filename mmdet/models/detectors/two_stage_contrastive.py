from mmcv.runner.fp16_utils import auto_fp16
from numpy.lib.arraysetops import isin
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
                 pretrained=None,
                 same_scale=True):
        super(TwoStageDetectorForContrastive, self).__init__()
        self.backbone = build_backbone(backbone)

        rand_box.update(dict(same_scale=same_scale))
        self.gen_rand_box = build_rand_box(rand_box)

        self.contrastive_head = build_contrastive_head(contrastive_head)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.same_scale = same_scale

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

    def forward_train(self, img, img_metas,
                      img_2=None, img_metas_2=None, **kwargs):
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

        losses = dict()

        if self.same_scale:
            rand_box_input = dict(
                img=img,
                img_metas=img_metas)

            pseudo_gt_bboxes_1, pseudo_gt_bboxes_2, \
                img_1, img_2 = self.gen_rand_box(rand_box_input)

            roi_feats_1 = self.forward_each_img(
                img_1, img_metas, 1, pseudo_gt_bboxes_1, losses)
            roi_feats_2 = self.forward_each_img(
                img_2, img_metas, 2, pseudo_gt_bboxes_2, losses)

        else:
            assert img_2 is not None and img_metas_2 is not None

            img_1 = img
            img_metas_1 = img_metas

            rand_box_input = dict(
                img_1=img_1,
                img_metas_1=img_metas_1,
                img_2=img_2,
                img_metas_2=img_metas_2)

            pseudo_gt_bboxes_1, pseudo_gt_bboxes_2, \
                img_1, img_2 = self.gen_rand_box(rand_box_input)

            roi_feats_1 = self.forward_each_img(
                img_1, img_metas_1, 1, pseudo_gt_bboxes_1, losses)
            roi_feats_2 = self.forward_each_img(
                img_2, img_metas_2, 2, pseudo_gt_bboxes_2, losses)

        # forward projection, prediction and loss
        constrastive_losses = self.contrastive_head(roi_feats_1,
                                                    roi_feats_2)
        losses.update(constrastive_losses)
        return losses

    def forward_each_img(self, img, img_metas, img_id,
                         pseudo_gt_bboxes, losses=None):
        assert isinstance(img_id, int)

        x = self.extract_feat(img)

        # RPN forward and loss
        if self.with_rpn:
            rpn_losses = self.rpn_head.forward_train(
                x,
                img_metas,
                pseudo_gt_bboxes)
            rpn_losses = self.identify(rpn_losses, img_id)
            losses.update(rpn_losses)

        roi_feats = self.roi_head.forward_train(x, pseudo_gt_bboxes)

        return roi_feats

    def train_step(self, data, optimizer):
        losses = self(data)

        loss, log_vars = self._parse_losses(losses)

        if not isinstance(data, list):
            data = [data]
        outputs = dict(loss=loss, log_vars=log_vars,
                       num_samples=len(data[0]['img_metas']))

        return outputs

    def forward(self, data, **kwargs):
        if self.same_scale:
            return self.forward_same_scale(**data)
        else:
            input_data = dict(
                img=data[0]['img'],
                img_metas=data[0]['img_metas'],
                img_2=data[1]['img'],
                img_metas_2=data[1]['img_metas'])
            return self.forward_diff_scale(**input_data)

    @auto_fp16(apply_to=('img', ))
    def forward_same_scale(self, img, img_metas, **kwargs):
        return self.forward_train(img, img_metas, **kwargs)

    @auto_fp16(apply_to=('img_1', 'img_2'))
    def forward_diff_scale(self,
                           img, img_metas,
                           img_2, img_metas_2, **kwargs):
        return self.forward_train(img, img_metas,
                                  img_2, img_metas_2, **kwargs)

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
