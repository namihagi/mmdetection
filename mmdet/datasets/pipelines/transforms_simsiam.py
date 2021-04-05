import cv2
import mmcv
import numpy as np
from copy import deepcopy
from numpy import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms.transforms import RandomHorizontalFlip

from ..builder import PIPELINES
from .transforms_contrastive import GaussianBlur


@PIPELINES.register_module()
class SimsiamAugWithCrop(object):

    def __init__(self,
                 to_rgb=True,
                 jitter_param=dict(
                     brightness=0.4,
                     contrast=0.4,
                     saturation=0.4,
                     hue=0.1),
                 jitter_p=0.8,
                 grayscale_p=0.2,
                 gaussian_sigma=[0.1, 2.0],
                 gaussian_p=0.5,
                 resize_size=224,
                 resize_scale=(0.2, 1.0)):

        self.to_rgb = to_rgb
        self.resize_size = resize_size
        self.resize_scale = resize_scale

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(
                size=self.resize_size,
                scale=self.resize_scale),
            transforms.RandomHorizontalFlip(),
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

        # prepare 2 views
        img_1 = img_1.astype(np.uint8)
        img_pil_1 = Image.fromarray(img_1, mode="RGB")
        img_pil_2 = deepcopy(img_pil_1)

        # augmentation
        img_1 = self.augment_each_img(img_pil_1, dtype)
        img_2 = self.augment_each_img(img_pil_2, dtype)

        # concat images
        img_concat = np.concatenate([img_1, img_2], axis=-1)

        results['img'] = img_concat
        results['img_shape'] = (self.resize_size, self.resize_size, 3)
        return results

    def augment_each_img(self, img_pil, dtype):
        img_pil = self.augmentation(img_pil)
        img = np.array(img_pil, dtype=dtype)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
