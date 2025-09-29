import copy
import random
from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch
try:
    import imgaug.augmenters as iaa
except:
    print("imgaug import error, possibly caused by using numpy >= 2.x.x")
    iaa = None

from pathlib import Path
from PIL import Image
from typing import List
import math
from torchvision import transforms

from hiad.utils.split_and_gather import LRPatch
from hiad.syn.utils import rand_perlin_2d_np


class BaseAnomalySynthesizer(ABC):

    @abstractmethod
    def anomaly_synthesize(self, sample: LRPatch, **kwargs) -> LRPatch:
        raise NotImplementedError

    def copy_low_resolution_images(self, dst_sample: LRPatch, src_sample: LRPatch):
        if src_sample.low_resolution_images is None:
            return dst_sample
        else:
            for low_resolution_image, low_resolution_index in zip(src_sample.low_resolution_images, src_sample.low_resolution_indexes):
                new_low_resolution_image = copy.deepcopy(low_resolution_image)
                new_low_resolution_image[low_resolution_index.y: low_resolution_index.y + low_resolution_index.height,
                                         low_resolution_index.x: low_resolution_index.x + low_resolution_index.width, :] = \
                    cv2.resize(dst_sample.image, (low_resolution_index.width, low_resolution_index.height))

                if dst_sample.low_resolution_indexes is None:
                    dst_sample.low_resolution_indexes = []
                if dst_sample.low_resolution_images is None:
                    dst_sample.low_resolution_images = []

                dst_sample.low_resolution_images.append(new_low_resolution_image)
                dst_sample.low_resolution_indexes.append(copy.deepcopy(low_resolution_index))
            return dst_sample


class ImageBlendingSynthesizer(BaseAnomalySynthesizer):
    def __init__(self,
                 p: float,
                 anomaly_source: str,
                 perlin_scale = 6,
                 min_perlin_scale = 0,
                 perlin_noise_threshold = 0.5,
                 transparency_range = None):

        assert p >= 0 and p <= 1
        self.p = p
        self.anomaly_source = anomaly_source
        self.min_perlin_scale = min_perlin_scale
        self.perlin_scale = perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold

        self.transparency_range = transparency_range if transparency_range is not None else [0.5, 1.0]
        path = Path(self.anomaly_source)
        self.anomaly_files = list(path.rglob('*.png')) + list(path.rglob('*.jpg'))
        assert len(self.anomaly_files) != 0, "The anomaly source is empty. Please correctly configure the path to the anomaly source dataset, such as data/DTD."

    def rotate_image_cv2(self, image, angle_range=(-90, 90)):
        angle = random.uniform(*angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        return rotated

    def generate_perlin_noise_mask(self, image_shape) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np((image_shape[0], image_shape[1]), (perlin_scalex, perlin_scaley))

        perlin_noise = self.rotate_image_cv2(perlin_noise, angle_range=(-90, 90))

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )
        return mask_noise

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        return aug

    def anomaly_synthesize(self, sample: LRPatch, **kwargs) -> LRPatch:
        assert sample.label is None or sample.label == 0
        if random.random() < self.p:
            image = sample.image.astype(np.float32)
            while True:
                mask = self.generate_perlin_noise_mask(image.shape[:2])
                if np.max(mask) != 0:
                    break
            idx = np.random.choice(len(self.anomaly_files))
            anomaly_source_img = np.array(Image.open(self.anomaly_files[idx]).convert('RGB'))
            anomaly_source_img = cv2.resize(anomaly_source_img, dsize=image.shape[:2])

            if iaa is not None:
                anomaly_source_img = self.rand_augment()(image=anomaly_source_img).astype(np.float32)
            else:
                anomaly_source_img = anomaly_source_img.astype(np.float32)

            factor = np.random.uniform(*self.transparency_range, size=1)[0]
            mask_expanded = np.expand_dims(mask, axis=2)
            anomaly_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * image)
            anomaly_img = ((- mask_expanded + 1) * image) + anomaly_img
            mask = mask.astype(np.uint8)
            mask[mask != 0] = 1

            anomaly_sample = LRPatch(image=anomaly_img.astype(np.uint8),
                     mask = mask,
                     foreground=sample.foreground,
                     label = 1,
                     label_name=sample.label_name,
                     clsname = sample.clsname,
                     main_index = sample.main_index)

            return self.copy_low_resolution_images(anomaly_sample, sample)
        else:
            sample = copy.deepcopy(sample)
            if sample.label is None:
                sample.label = 0
            return sample


class RandomeBoxSynthesizer(BaseAnomalySynthesizer):

    def __init__(self,
                 p: float,
                 max_patch_num: int = None,
                 anomaly_sizes: List = None,
                 diff_threshold: List = None):

        assert p >= 0 and p <= 1
        self.p = p
        self.max_patch_num = max_patch_num if max_patch_num is not None else 5
        self.anomaly_sizes = anomaly_sizes if anomaly_sizes is not None else [[16, 40], [80, 100]]
        if np.array(self.anomaly_sizes).ndim == 1:
            self.anomaly_sizes = [self.anomaly_sizes]
        self.diff_threshold = diff_threshold if diff_threshold is not None else [20, 60]

    def anomaly_synthesize(self, sample: LRPatch, **kwargs) -> LRPatch:

        assert sample.label is None or sample.label == 0
        if random.random() < self.p:
            image = sample.image.astype(np.float32)

            mask = np.zeros_like(image[..., 0:1])  # single channel
            patchex = image.copy()

            coor_min_dim1, coor_max_dim1, coor_min_dim2, coor_max_dim2 = mask.shape[0] - 1, 0, mask.shape[1] - 1, 0

            for i in range(self.max_patch_num):
                if i == 0 or np.random.randint(2) > 0:  # at least one patch
                    try:
                        patchex, (
                        (_coor_min_dim1, _coor_max_dim1), (_coor_min_dim2, _coor_max_dim2)), patch_mask = self._patch_ex(
                            patchex)
                    except:
                        continue

                    if patch_mask is not None:
                        mask[_coor_min_dim1:_coor_max_dim1, _coor_min_dim2:_coor_max_dim2] = patch_mask
                        coor_min_dim1 = min(coor_min_dim1, _coor_min_dim1)
                        coor_max_dim1 = max(coor_max_dim1, _coor_max_dim1)
                        coor_min_dim2 = min(coor_min_dim2, _coor_min_dim2)
                        coor_max_dim2 = max(coor_max_dim2, _coor_max_dim2)

            mask[mask != 0] = 1

            anomaly_sample = LRPatch(image = patchex.astype(np.uint8),
                                     mask = mask.squeeze(-1).astype(np.uint8),
                                     foreground = sample.foreground,
                                     label = 1,
                                     label_name = sample.label_name,
                                     clsname = sample.clsname,
                                     main_index = sample.main_index)

            return self.copy_low_resolution_images(anomaly_sample, sample)
        else:
            sample = copy.deepcopy(sample)
            if sample.label is None:
                sample.label = 0
            return sample


    def _patch_ex(self, ima_dest):
        dims = np.array(ima_dest.shape)
        anomaly_sizes = random.choice(self.anomaly_sizes)
        height = int(random.uniform(*anomaly_sizes))
        width = int(random.uniform(*anomaly_sizes))
        ok = False
        attempts = 0
        while not ok:
            pixel = random.randint(0, 255)
            src = (np.ones((height, width, 3)) * pixel).astype(np.uint8)
            patch_mask = (np.ones((height, width, 1))).astype(np.uint8)
            center_dim1 = np.random.randint(height // 2 + 1, ima_dest.shape[0] - height // 2 - 1)
            center_dim2 = np.random.randint(width // 2 + 1, ima_dest.shape[1] - width // 2 - 1)
            coor_min_dim1, coor_max_dim1 = center_dim1 - height // 2, center_dim1 + (height + 1) // 2
            coor_min_dim2, coor_max_dim2 = center_dim2 - width // 2, center_dim2 + (width + 1) // 2
            diff = np.mean(np.abs(ima_dest[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] - src))
            ok = (diff >= self.diff_threshold[0] and diff <= self.diff_threshold[1])
            attempts += 1
            if attempts == 50:
                raise RuntimeError
        patchex = ima_dest.copy()
        patchex[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] = src
        return patchex.astype(np.uint8), ((coor_min_dim1, coor_max_dim1), (coor_min_dim2, coor_max_dim2)), patch_mask



class CutPasteAnomalySynthesizer(BaseAnomalySynthesizer):

    def __init__(self,
                 p: float,
                 anomaly_sizes: List = None,
                 aspect_ratio: float = 0.3
                 ):

        assert p >= 0 and p <= 1
        self.p = p
        self.anomaly_sizes = anomaly_sizes if anomaly_sizes is not None else [[16, 40], [80, 100]]
        if np.array(self.anomaly_sizes).ndim==1:
            self.anomaly_sizes = [self.anomaly_sizes]
        self.aspect_ratio = aspect_ratio

    def anomaly_synthesize(self, sample: LRPatch, **kwargs) -> LRPatch:
        assert sample.label is None or sample.label == 0
        if random.random() < self.p:
            image = sample.image.astype(np.float32)
            patchex, mask = self.CutPasteNormal(image)
            anomaly_sample = LRPatch(image=patchex.astype(np.uint8),
                     mask = mask,
                     foreground = sample.foreground,
                     label = 1,
                     label_name = sample.label_name,
                     clsname = sample.clsname,
                     main_index = sample.main_index)
            return self.copy_low_resolution_images(anomaly_sample, sample)
        else:
            sample = copy.deepcopy(sample)
            if sample.label is None:
                sample.label = 0
            return sample

    def CutPasteNormal(self, image):
        h = image.shape[0]
        w = image.shape[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(*random.choice(self.anomaly_sizes))**2

        log_ratio = np.log(np.array([self.aspect_ratio, 1 / self.aspect_ratio]))

        aspect = np.exp(
            np.random.uniform(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = image[from_location_h:from_location_h + cut_h, from_location_w:from_location_w + cut_w,: ]

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        augmented = image.copy()
        augmented[to_location_h:to_location_h + cut_h, to_location_w:to_location_w + cut_w,: ] = box

        mask = np.zeros((h, w))
        mask[to_location_h:to_location_h + cut_h, to_location_w:to_location_w + cut_w] = 1
        return augmented, mask.astype(np.uint8)



class ColorShiftSynthesizer(BaseAnomalySynthesizer):

    def __init__(self, p: float):

        assert p >= 0 and p <= 1
        self.p = p
        self.transform_ae = transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2)
        ])

    def anomaly_synthesize(self, sample: LRPatch, **kwargs) -> LRPatch:
        assert sample.label is None or sample.label == 0
        if random.random() < self.p:
            image = sample.image.astype(np.float32)
            aug_image = self.transform_ae(Image.fromarray(image.astype(np.uint8)))
            anomaly_sample = LRPatch(image=np.array(aug_image).astype(np.uint8),
                     foreground = sample.foreground,
                     label = 1,
                     label_name = sample.label_name,
                     clsname = sample.clsname,
                     main_index = sample.main_index)
            return self.copy_low_resolution_images(anomaly_sample, sample)
        else:
            sample = copy.deepcopy(sample)
            if sample.label is None:
                sample.label = 0
            return sample

