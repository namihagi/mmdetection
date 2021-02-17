import torch.nn as nn
from mmcv.runner import auto_fp16
from torch.nn.modules.utils import _pair

from mmdet.models.builder import HEADS


@HEADS.register_module()
class BBoxHeadForContrastive(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 roi_feat_size=7,
                 in_channels=256):
        super(BBoxHeadForContrastive, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        self.debug_imgs = None

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def init_weights(self):
        pass
