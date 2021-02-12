import inspect

import mmcv
import numpy as np
from numpy import random

from copy import deepcopy
from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES

from .transforms import Resize, Normalize, Pad

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class ResizeForContrastive(Resize):

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img_1 = deepcopy(results[key][:, :, :3])
                img_1, scale_factor = mmcv.imrescale(
                    img_1,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                img_2 = deepcopy(results[key][:, :, 3:])
                img_2, scale_factor = mmcv.imrescale(
                    img_2,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img_1.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img_1 = deepcopy(results[key][:, :, :3])
                img_1, w_scale, h_scale = mmcv.imresize(
                    img_1,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                img_2 = deepcopy(results[key][:, :, :3])
                img_2, w_scale, h_scale = mmcv.imresize(
                    img_2,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            img = np.concatenate([img_1, img_2], axis=-1)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img[:, :, :3].shape
            # in case that there is no padding
            results['pad_shape'] = img[:, :, :3].shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio


@PIPELINES.register_module()
class PhotoMetricDistortionForContrastive(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        img_2 = deepcopy(img)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta
        if random.randint(2):
            delta_2 = random.uniform(-self.brightness_delta,
                                     self.brightness_delta)
            img_2 += delta_2

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha
        mode_2 = random.randint(2)
        if mode_2 == 1:
            if random.randint(2):
                alpha_2 = random.uniform(self.contrast_lower,
                                         self.contrast_upper)
                img_2 *= alpha_2

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)
        img_2 = mmcv.bgr2hsv(img_2)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)
        if random.randint(2):
            img_2[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
        if random.randint(2):
            img_2[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img_2[..., 0][img_2[..., 0] > 360] -= 360
            img_2[..., 0][img_2[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)
        img_2 = mmcv.hsv2bgr(img_2)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha
        if mode_2 == 0:
            if random.randint(2):
                alpha_2 = random.uniform(self.contrast_lower,
                                         self.contrast_upper)
                img_2 *= alpha_2

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]
        if random.randint(2):
            img_2 = img_2[..., random.permutation(3)]

        # concat images
        img_concat = np.concatenate([img, img_2], axis=-1)

        img_fig = img - img.min()
        img_fig = img_fig / img_fig.max()
        img_fig = np.clip(img_fig, 0, 1)
        img_fig_2 = img_2 - img_2.min()
        img_fig_2 = img_fig_2 / img_fig_2.max()
        img_fig_2 = np.clip(img_fig_2, 0, 1)

        results['img'] = img_concat
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class NormalizeForContrastive(Normalize):

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            img_1 = deepcopy(results[key][:, :, :3])
            img_1 = mmcv.imnormalize(img_1, self.mean, self.std,
                                     self.to_rgb)
            img_2 = deepcopy(results[key][:, :, 3:])
            img_2 = mmcv.imnormalize(img_2, self.mean, self.std,
                                     self.to_rgb)
            img = np.concatenate([img_1, img_2], axis=-1)
            results[key] = img
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class PadForContrastive(Pad):

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img[:, :, :3].shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
