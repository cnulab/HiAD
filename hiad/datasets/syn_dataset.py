from typing import List
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy

from hiad.syn import BaseAnomalySynthesizer
from hiad.utils.split_and_gather import LRPatch


class AnomalySynDataset(Dataset):

    def __init__(
        self,
        patches: List[LRPatch],
        synthesizer: BaseAnomalySynthesizer,
        training: bool,
        task_name: str,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
        **kwargs,
    ):

        self.patches = patches
        self.synthesizer = synthesizer
        self.training = training
        self.task_name = task_name

        if normalize_mean is None:
            normalize_mean = [0.485, 0.456, 0.406]

        if normalize_std is None:
            normalize_std = [0.229, 0.224, 0.225]

        self.transform_img = [
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]

        self.transform_mask = [
            transforms.ToTensor()
        ]

        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_mask = transforms.Compose(self.transform_mask)
        super().__init__()


    def __getitem__(self, idx):
        items = {}
        patch = self.patches[idx]
        image = patch.image.astype(numpy.uint8)
        image = self.transform_img(image)
        items.update({'gt_image': image})

        if patch.low_resolution_images is not None:
            for i, (low_resolution_image,  low_resolution_index) in enumerate(
                    zip(patch.low_resolution_images, patch.low_resolution_indexes)):
                low_resolution_image = low_resolution_image.astype(numpy.uint8)
                low_resolution_image = self.transform_img(low_resolution_image)
                items[f'gt_low_resolution_image_{i}'] = low_resolution_image
                items[f'gt_low_resolution_index_{i}'] = str(low_resolution_index)

        anomaly_patch = self.synthesizer.anomaly_synthesize(patch)
        image = self.transform_img(anomaly_patch.image.astype(numpy.uint8))
        items.update({'image': image})

        if anomaly_patch.low_resolution_images is not None:
            for i, (low_resolution_image, low_resolution_index) in enumerate(
                    zip(anomaly_patch.low_resolution_images, anomaly_patch.low_resolution_indexes)):
                low_resolution_image = low_resolution_image.astype(numpy.uint8)
                low_resolution_image = self.transform_img(low_resolution_image)
                items[f'low_resolution_image_{i}'] = low_resolution_image
                items[f'low_resolution_index_{i}'] = str(low_resolution_index)

        if anomaly_patch.mask is not None:
            mask = self.transform_mask(anomaly_patch.mask.astype(numpy.uint8)).squeeze(0)
            mask [mask!=0] = 1
        else:
            mask = torch.zeros(image.shape[1:])

        items.update({
            "mask": mask
        })

        if patch.clsname is not None:
            items.update({"clsname": patch.clsname})

        if patch.label is not None:
            items.update({"label": patch.label})

        return items

    def __len__(self):
        return len(self.patches)