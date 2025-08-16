from abc import ABC, abstractmethod
from typing import List
import numpy as np
from torch.utils.data import DataLoader
import numpy
import torch
import torch.nn.functional as F
import cv2
import scipy.ndimage as ndimage
import copy
from tqdm import tqdm

from hiad.utils.base import get_avg_score
from hiad.utils.split_and_gather import HRImageIndex, LRPatch
from hiad.datasets.patch_dataset import PatchDataset

class BaseDetector(ABC):

    def __init__(self, patch_size, device, fusion_weights = None, logger = None, seed =0, early_stop_epochs = -1, **kwargs):
        r"""
           Base class for detectors. New detectors can be created by inheriting from this class.
           Args:
               patch_size (int or list): Resolution of image patches.
               device (torch.Device): Model load device.
               fusion_weights (list): Fusion weights for feature fusion. If None, a set of equal fusion weights will be used.
               logger (logging.Logger): Logger object
               seed (int): random seed.
               early_stop_epochs (int): Controls early stopping for detector: if a detector shows no performance improvement on the validation set for N epochs, training will be stopped.
                            If -1, Early stopping will be disabled.
           """

        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        else:
            self.patch_size = patch_size
        self.seed = seed
        self.logger = logger
        self.device = device
        self.early_stop_epochs = early_stop_epochs
        self.early_stop_cur = self.early_stop_epochs
        self.fusion_weights = fusion_weights

    @abstractmethod
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        r"""
           This method encodes image patches into feature representations (feature extraction).

           Args
               input_tensor (torch.Tensor): Input image patch tensor. Shape: (B,3,H,W)
           return:
                Returns a list of extracted features (multi-scale features). Shape: ([B,C1,H1,W1], [B,C2,H2,W2], ..., [B,Cn,Hn,Wn])
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators=None,
                   ) -> bool:
        r"""
           This method defines the training procedure of the model.

           Args
               train_dataloader (torch.utils.data.DataLoader): DataLoader used for training; the format of the returned data is defined by `create_dataset`.
               task_name (str): Task Name.
               checkpoint_path: Path for saving model checkpoints.
               val_dataloader (torch.utils.data.DataLoader): DataLoader used for validation; the format of the returned data is defined by `create_dataset`.
                    If no validation configuration is provided, this value will be None.
               evaluators: Evaluation method used for validation.
               The `val_step` method provides a built-in implementation of the validation procedure.
               You can perform validation using the code below; `best_metrics` include the historically best validation scores.
                    best_metrics = {}
                    best_metrics = self.val_step(val_dataloader, evaluators, checkpoint_path, best_metrics)

           return:
                Returns a boolean: True if the checkpoint has been saved; False if not, which will trigger the `trainer` to perform the save.
        """
        raise NotImplementedError


    @abstractmethod
    def inference_step(self,
                       test_dataloader: DataLoader,
                       task_name: str):
        r"""
           This method defines the inference (test) procedure of the model.

           Args
               test_dataloader (torch.utils.data.DataLoader): DataLoader used for testing; the format of the returned data is defined by `create_dataset`.
               task_name (str): Task Name.

           return:
                Returns a list of numpy.ndarray, where each array corresponds to the detection results of an individual sample.
                Shape: ([H,W],..., [H,W]), where H and W is the height and width of the image.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self,
                        checkpoint_path: str
                        ):
        r"""
            save checkpoint
            Args
                checkpoint_path (str): Path of checkpoint
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        r"""
            load checkpoint
            Args
                checkpoint_path (str): Path of checkpoint
        """
        raise NotImplementedError

    @torch.no_grad()
    def val_step(self,
                 val_dataloader,
                 evaluators,
                 checkpoint_path,
                 best_metrics):
        r"""
                This method defines the basic validation procedure.

                Args
                    val_dataloader (torch.utils.data.DataLoader): DataLoader used for validation; the format of the returned data is defined by `create_dataset`.
                    evaluators: Evaluation method used for validation.
                    checkpoint_path: Path for saving model checkpoints.
                    best_metrics: the best historical metrics

                return:
                    best_metrics: updated best metrics
        """

        pred_masks = self.inference_step(val_dataloader, task_name='val')
        pred_masks = numpy.stack(pred_masks)
        self.max_anomaly_score, self.min_anomaly_score = numpy.max(pred_masks), numpy.min(pred_masks)
        pred_masks = self.patch_post_processing(pred_masks)
        gt_labels = [patch.label for patch in val_dataloader.dataset.patches]
        gt_masks = []
        for patch in val_dataloader.dataset.patches:
            if patch.mask is not None:
                mask = copy.deepcopy(patch.mask)
                mask[mask != 0] = 1
            else:
                mask = numpy.zeros(pred_masks.shape[1:]).astype(float)
            gt_masks.append(mask)
        gt_masks = numpy.stack(gt_masks)
        pred_labels = self.get_image_score(pred_masks)
        scores = {}
        for evaluator_fn in evaluators:
            scores.update(evaluator_fn(prediction_masks=pred_masks,
                                       gt_masks=gt_masks,
                                       prediction_scores=pred_labels,
                                       gt_labels=gt_labels,
                                       device = self.device))

        for key in scores:
            self.logger.info(f'{key} is: {scores[key]:.4f}')

        if best_metrics == {}:
            best_metrics = scores
            for key in scores:
                self.logger.info(f'--->new best {key} is: {scores[key]:.4f}')
            self.save_checkpoint(checkpoint_path)
        else:
            if 'image_auroc' in scores:
                if scores['image_auroc'] > best_metrics['image_auroc']:
                    best_metrics = scores
                    for key in scores:
                        self.logger.info(f'--->new best {key} is: {scores[key]:.4f}')
                    self.save_checkpoint(checkpoint_path)
                    self.early_stop_cur = self.early_stop_epochs
                elif scores['image_auroc'] == best_metrics['image_auroc'] and get_avg_score(scores) > get_avg_score(best_metrics):
                    best_metrics = scores
                    for key in scores:
                        self.logger.info(f'--->new best {key} is: {scores[key]:.4f}')
                    self.save_checkpoint(checkpoint_path)
                    self.early_stop_cur = self.early_stop_epochs
                else:
                    self.early_stop_cur -= 1
            else:
                if get_avg_score(scores) > get_avg_score(best_metrics):
                    best_metrics = scores
                    for key in scores:
                        self.logger.info(f'--->new best {key} is: {scores[key]:.4f}')
                    self.save_checkpoint(checkpoint_path)
                    self.early_stop_cur = self.early_stop_epochs
                else:
                    self.early_stop_cur -= 1
        if self.early_stop_cur >= 0:
            self.logger.info(f'Early stop cur is: {self.early_stop_cur}')
            if self.early_stop_cur == 0:
                self.logger.info(f'Early stop !!')
        return best_metrics


    @staticmethod
    def post_processing(patch_segmentations: numpy.ndarray,
                        thumbnail_segmentations: numpy.ndarray = None,
                        is_normalize = False, gaussian_filter = True):
        r"""
                This method aggregates high-resolution and low-resolution detection results and performs post-processing.
                Args
                    patch_segmentations (numpy.ndarray): High-resolution detection scores. Shape [B,H,W]
                    thumbnail_segmentations (numpy.ndarray): Low-resolution detection scores. Shape [B,H_low,W_low]
                    is_normalize: Normalize or not
                    gaussian_filter: Gaussian Filter or not
                return:
                    FINAL detection scores. Shape [B,H,W]
        """

        if thumbnail_segmentations is not None:
            thumbnail_segmentations = F.interpolate(torch.tensor(thumbnail_segmentations).unsqueeze(1),
                            size=(patch_segmentations.shape[1], patch_segmentations.shape[2]),
                            mode="bilinear",
                            align_corners=False).squeeze(1).numpy()
            segmentations = np.where(patch_segmentations >= thumbnail_segmentations,
                                     patch_segmentations,
                                     thumbnail_segmentations)
        else:
            segmentations = patch_segmentations

        if is_normalize:
            segmentations = cv2.normalize(segmentations, None, 0, 1, cv2.NORM_MINMAX)

        if not gaussian_filter:
            return segmentations
        else:
            segmentations = [ndimage.gaussian_filter(segmentation, sigma = 4) for segmentation in tqdm(segmentations, desc = 'Gaussian Filter')]
            return np.stack(segmentations)


    def patch_post_processing(self, patch_anomaly_score, eps = 1e-4):
        if self.min_anomaly_score is None or self.max_anomaly_score is None:
            return patch_anomaly_score
        patch_anomaly_score = (patch_anomaly_score - self.min_anomaly_score) / (
                    self.max_anomaly_score - self.min_anomaly_score + eps)
        return numpy.clip(patch_anomaly_score, 0, 1)


    def create_dataset(self, patches: List[LRPatch], training: bool, task_name: str):

        r"""
                This method defines how data is loaded for the model by constructing a `Dataset` to create a `DataLoader` object.
                Args
                    patches (List[LRPatch]): A set of low-resolution Patch objects.
                    training (boolean): True if training, else False.
                    task_name (str): Task Name
                return:
                    return a torch.utils.data.Dataset object
        """
        dataset = PatchDataset(patches = patches, training = training, task_name = task_name)
        return dataset

    @torch.no_grad()
    def get_multi_resolution_fusion_embeddings(self, data) -> List[torch.Tensor]:
        r"""
            This method can obtain multi-resolution fusion features for patches. It is only available when `embedding` method is defined.
            Args:
                data: The data produced by iterating over the `DataLoader`;
                    its specific format is defined by the `Dataset` created via `create_dataset` method.
            return:
                Returns a list of fusion features (multi-scale features). Shape: ([B,C1,H1,W1], [B,C2,H2,W2], ..., [B,Cn,Hn,Wn]) same as 'embedding' method

        """
        image = data['image'].to(self.device)
        embeddings = self.embedding(image)
        low_resolution_image_keys = [key for key in data if key.startswith('low_resolution_image')]

        if len(low_resolution_image_keys) == 0:
            return embeddings
        else:
            if self.fusion_weights is not None:
                assert len(self.fusion_weights) == len(low_resolution_image_keys) + 1, "fusion_weights must be the same length as ds_factor."
                fusion_weights = [weight / sum(self.fusion_weights) for weight in self.fusion_weights]
            else:
                fusion_weights = [1 / (len(low_resolution_image_keys) + 1)] * (len(low_resolution_image_keys)+1)

            embeddings = [[embedding * fusion_weights[0]] for embedding in embeddings]

            low_resolution_image_keys.sort(key = lambda item: int(item.split('_')[-1]))

            for rs_index, low_resolution_image_key in enumerate(low_resolution_image_keys):
                low_resolution_image = data[low_resolution_image_key].to(self.device)
                low_resolution_index = data[low_resolution_image_key.replace('image', 'index')]
                low_resolution_embeddings = self.embedding(low_resolution_image)

                for i, low_resolution_embedding in enumerate(low_resolution_embeddings):
                    downsampling_embedding = []
                    for feature, index in zip(low_resolution_embedding, low_resolution_index):
                        feature_stride_H, feature_stride_W = self.patch_size[1] / feature.shape[1], self.patch_size[0] / feature.shape[2]
                        index = HRImageIndex.from_str(index)
                        x_start = index.x / feature_stride_W
                        y_start = index.y / feature_stride_H
                        x_end = x_start + index.width / feature_stride_W
                        y_end = y_start + index.height / feature_stride_H
                        downsampling_embedding.append(feature[:, int(y_start) : int(y_end), int(x_start) : int(x_end)])
                    try:
                        downsampling_embedding = torch.stack(downsampling_embedding)
                    except:
                        first_embedding = downsampling_embedding[0]
                        downsampling_embedding = [F.interpolate(feat.unsqueeze(0), size=(first_embedding.shape[-2], first_embedding.shape[-1]), mode='bilinear').squeeze(0)
                                                  for feat in downsampling_embedding[1:]]

                        downsampling_embedding = [first_embedding] + downsampling_embedding
                        downsampling_embedding = torch.stack(downsampling_embedding)

                    embeddings[i].append(fusion_weights[rs_index+1] * F.interpolate(
                            downsampling_embedding,
                            size=(embeddings[i][-1].shape[-2], embeddings[i][-1].shape[-1]),
                            mode="bilinear",
                            align_corners=False,
                            ))

            embeddings = [torch.sum(torch.stack(embedding), dim=0, keepdim=False) for embedding in embeddings]
            return embeddings

    def feature_dimensions(self):
        _input = torch.ones([1, 3, self.patch_size[1], self.patch_size[0]]).to(self.device)
        embeddings = self.embedding(_input)
        return [embedding.shape[1] for embedding in embeddings]

    @staticmethod
    def get_image_score(segmentations, pool_size=16, batch_size = 32):
        r"""
            This method defines the process of obtaining image-level anomaly detection scores.
            Args
                segmentations (torch.Tensor): pixel-level detection scores. Shape: [B,H,W]
            return:
                return image-level detection score. Shape: [B]
        """
        N, H, W = segmentations.shape
        preds = torch.tensor(segmentations[:, None, ...])
        preds = preds.split(batch_size, dim=0)
        preds = [(
            F.avg_pool2d(pred.cuda(), (pool_size, pool_size), stride=1).cpu().numpy()
        ) for pred in preds]
        preds = np.concatenate(preds, axis=0)
        return preds.reshape(N, -1).max(axis=1)