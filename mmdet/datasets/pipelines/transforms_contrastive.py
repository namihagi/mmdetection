import cv2
import mmcv
import numpy as np
from copy import deepcopy
from numpy import random
from PIL import Image, ImageFilter
from torchvision import transforms

from ..builder import PIPELINES
from .transforms import Normalize, Pad, RandomFlip, Resize


@PIPELINES.register_module()
class ResizeForContrastive(Resize):

    def __init__(self, same_scale=True, **kwarg):
        super(ResizeForContrastive, self).__init__(**kwarg)

        self.same_scale = same_scale

    def _resize_img(self, results):
        if self.same_scale:
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

        else:
            super()._resize_img(results)

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


@PIPELINES.register_module()
class RandomFlipForContrastive(RandomFlip):

    def __init__(self, same_scale=True, **kwarg):
        super(RandomFlipForContrastive, self).__init__(**kwarg)

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
            flip_dict = dict(
                flip=results_1['flip'],
                flip_direction=results_1['flip_direction'])
            results_2.update(flip_dict)
            results_2 = super().__call__(results_2)

            return tuple([results_1, results_2])


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
        img_1 = results['img']
        img_2 = deepcopy(img_1)
        assert img_1.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'

        img_1 = self.augment(img_1)
        img_2 = self.augment(img_2)

        # concat images
        img_concat = np.concatenate([img_1, img_2], axis=-1)

        results['img'] = img_concat
        return results

    def augment(self, img):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img

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

    def __init__(self, same_scale=True, **kwarg):
        super(NormalizeForContrastive, self).__init__(**kwarg)

        self.same_scale = same_scale

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        if self.same_scale:
            assert not isinstance(results, tuple)

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

        else:
            assert isinstance(results, tuple)
            results_1 = results[0]
            results_2 = results[1]

            results_1 = super().__call__(results_1)
            results_2 = super().__call__(results_2)

            return tuple([results_1, results_2])


@PIPELINES.register_module()
class PadForContrastive(Pad):

    def __init__(self, same_scale=True, **kwarg):
        super(PadForContrastive, self).__init__(**kwarg)

        self.same_scale = same_scale

    def _pad_img(self, results):
        if self.same_scale:
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

        else:
            super()._pad_img(results)

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


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@PIPELINES.register_module()
class SimsiamAugmentation(object):

    def __init__(self,
                 to_rgb=True,
                 same_scale=True,
                 jitter_param=dict(
                     brightness=0.4,
                     contrast=0.4,
                     saturation=0.4,
                     hue=0.4),
                 jitter_p=0.8,
                 grayscale_p=0.2,
                 gaussian_sigma=[0.1, 2.0],
                 gaussian_p=0.5):

        self.to_rgb = to_rgb
        self.same_scale = same_scale

        self.augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(**jitter_param)
            ], p=jitter_p),
            transforms.RandomGrayscale(p=grayscale_p),
            transforms.RandomApply([
                GaussianBlur(sigma=gaussian_sigma)
            ], p=gaussian_p),
        ])

    def __call__(self, results):
        """Call function to perform byol augmentations on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'

        img_1 = results['img']
        dtype = img_1.dtype

        if self.to_rgb:
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

        img_1 = img_1.astype(np.uint8)
        img_pil_1 = Image.fromarray(img_1, mode="RGB")
        img_pil_2 = deepcopy(img_pil_1)

        img_1 = self.augment_each_img(img_pil_1, dtype)
        img_2 = self.augment_each_img(img_pil_2, dtype)

        if self.same_scale:
            # concat images
            img_concat = np.concatenate([img_1, img_2], axis=-1)

            results['img'] = img_concat
            return results

        else:
            results_1 = deepcopy(results)
            results['img_1'] = img_1

            results_2 = deepcopy(results)
            results['img_2'] = img_2

            return tuple([results_1, results_2])

    def augment_each_img(self, img_pil, dtype):
        img_pil = self.augmentation(img_pil)
        img = np.array(img_pil, dtype=dtype)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
