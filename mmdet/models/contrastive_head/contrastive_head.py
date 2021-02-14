"""
this code is refered to https://github.com/PatrickHua/SimSiam
"""

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.norm import build_norm_layer
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import CONTRASTIVE_HEAD, build_contrastive_head


@CONTRASTIVE_HEAD.register_module()
class ProjectionMLP(nn.Module):
    """Projection head for contrastive learning.

    Args:
        in_channels (int): Number of input channels for first layers.
        hidden_channels (int): Number of hidden layer channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True):
        super(ProjectionMLP, self).__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        if out_channels is None:
            out_channels = in_channels

        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            build_norm_layer(norm_cfg, hidden_channels)[1],
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            build_norm_layer(norm_cfg, hidden_channels)[1],
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            build_norm_layer(norm_cfg, out_channels)[1],
        )
        self.norm_eval = norm_eval

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def train(self, mode=True):
        super(ProjectionMLP, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@CONTRASTIVE_HEAD.register_module()
class PredictionMLP(nn.Module):
    """Prediction head for contrastive learning.

    Args:
        in_channels (int): Number of input channels for first layers.
        hidden_channels (int): Number of hidden layer channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True):
        super(PredictionMLP, self).__init__()

        if hidden_channels is None:
            hidden_channels = in_channels // 4

        if out_channels is None:
            out_channels = in_channels

        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            build_norm_layer(norm_cfg, hidden_channels)[1],
            nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(hidden_channels, out_channels)

        self.norm_eval = norm_eval

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def train(self, mode=True):
        super(PredictionMLP, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@CONTRASTIVE_HEAD.register_module()
class ContrastiveHead(nn.Module):
    """Contrasitve Head including projection and prediction."""

    def __init__(self,
                 projection,
                 prediction,
                 each_view_loss_weight=0.5):
        super(ContrastiveHead, self).__init__()

        self.projection = build_contrastive_head(projection)
        self.prediction = build_contrastive_head(prediction)

        self.each_view_loss_weight = each_view_loss_weight

    def forward(self, feat_1, feat_2):
        z_1, p_1 = self.proj_and_pred(feat_1)
        z_2, p_2 = self.proj_and_pred(feat_2)

        loss_1 = self.cosine_similarity(p_1, z_2)
        loss_2 = self.cosine_similarity(p_2, z_1)

        losses = dict(
            contrastive_loss_1=loss_1,
            contrastive_loss_2=loss_2)
        return losses

    def proj_and_pred(self, feat):
        z = self.projection(feat)
        p = self.prediction(z)
        return z, p

    def cosine_similarity(self, p, z):
        """
        args:
            p: a tensor of features (num_of_boxes, feat_dim)
            z: a tensor of features (num_of_boxes, feat_dim)
        return:
            negative cosine similarity between p and z
        """
        return -1.0 * self.each_view_loss_weight \
            * F.cosine_similarity(p, z.detach(), dim=-1).mean()
