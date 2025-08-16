from typing import List
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy

from hiad.utils.split_and_gather import LRPatch


class PatchDataset(Dataset):

    def __init__(
        self,
        patches: List[LRPatch],
        training: bool,
        task_name: str,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
        **kwargs,
    ):

        super().__init__()
        self.patches = patches
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

    def __getitem__(self, idx):

        patch = self.patches[idx]
        image = patch.image.astype(numpy.uint8)
        image = self.transform_img(image)

        if patch.mask is not None:
            mask = self.transform_mask(patch.mask.astype(numpy.uint8)).squeeze(0)
            mask [mask!=0] = 1
        else:
            mask = torch.zeros(image.shape[1:])

        item = {
            "image": image,
            "mask": mask,
        }

        if patch.clsname is not None:
            item.update({"clsname": patch.clsname})

        if patch.label is not None:
            item.update({"label": patch.label})

        if patch.low_resolution_images is not None:
            for i, (low_resolution_image, low_resolution_index) in enumerate(zip(patch.low_resolution_images, patch.low_resolution_indexes)):
                low_resolution_image = low_resolution_image.astype(numpy.uint8)
                low_resolution_image = self.transform_img(low_resolution_image)
                item[f'low_resolution_image_{i}'] = low_resolution_image
                item[f'low_resolution_index_{i}'] = str(low_resolution_index)
        return item

    def __len__(self):
        return len(self.patches)