import torch
import torch.distributed as dist
import torch.nn as nn
from collections import OrderedDict
from mmcv.runner.fp16_utils import auto_fp16

from ..builder import DETECTORS, build_backbone, build_contrastive_head


@DETECTORS.register_module()
class SimSiam(nn.Module):

    def __init__(self,
                 backbone,
                 contrastive_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SimSiam, self).__init__()

        self.backbone = build_backbone(backbone)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.contrastive_head = build_contrastive_head(contrastive_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        return x

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        img_1 = img[:, :3]
        img_2 = img[:, 3:]

        feat_1 = self.forward_each_img(img_1)
        feat_2 = self.forward_each_img(img_2)

        losses = self.contrastive_head(feat_1, feat_2)
        return losses

    def forward_each_img(self, img):

        x = self.extract_feat(img)
        assert len(x) == 1
        feat = self.gap(x[0])
        feat = torch.flatten(feat, start_dim=1)

        return feat

    def train_step(self, data, optimizer):
        losses = self(**data)

        loss, log_vars = self._parse_losses(losses)

        if not isinstance(data, list):
            data = [data]
        outputs = dict(loss=loss, log_vars=log_vars,
                       num_samples=len(data[0]['img_metas']))

        return outputs

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, **kwargs):
        return self.forward_train(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
