from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formating import Collect, DefaultFormatBundle


@PIPELINES.register_module()
class DefaultFormatBundleForContrastive(DefaultFormatBundle):

    def __init__(self, same_scale=True):
        super(DefaultFormatBundleForContrastive, self).__init__()

        self.same_scale = same_scale

    def __call__(self, results):
        if self.same_scale:
            assert not isinstance(results, tuple)
            return super().__call__(results)

        else:
            assert isinstance(results, tuple)
            results_1 = results[0]
            results_2 = results[1]

            results_1 = super().__call__(results_1)
            results_2 = super().__call__(results_2)

            return tuple([results_1, results_2])

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        if self.same_scale:
            img = results['img'][:, :, :3]
            img_2 = results['img'][:, :, 3:]
            assert img.shape == img_2.shape, "Error: shape not equal."
            results.setdefault('pad_shape', img.shape)
            results.setdefault('scale_factor', 1.0)
            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
            results.setdefault(
                'img_norm_cfg',
                dict(
                    mean=np.zeros(num_channels, dtype=np.float32),
                    std=np.ones(num_channels, dtype=np.float32),
                    to_rgb=False))
            return results

        else:
            return super()._add_default_meta_keys(results)


@PIPELINES.register_module()
class CollectForContrastive(Collect):

    def __init__(self, same_scale=True, **kwarg):
        super(CollectForContrastive, self).__init__(**kwarg)

        self.same_scale = same_scale

    def __call__(self, results):
        if self.same_scale:
            assert not isinstance(results, tuple)
            return super().__call__(results)

        else:
            assert isinstance(results, tuple)
            results_1 = results[0]
            results_2 = results[1]

            results_1 = super().__call__(results_1)
            results_2 = super().__call__(results_2)

            return tuple([results_1, results_2])
