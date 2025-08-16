from typing import List
import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import importlib
from timm.utils import dispatch_clip_grad
import torch.nn.functional as F

from hiad.datasets.patch_dataset import PatchDataset
from hiad.utils.split_and_gather import LRPatch, HRImageIndex
from .vitad.util.net import set_seed
from .vitad.model import get_model
from .vitad.optim import get_optim
from .vitad.optim.scheduler import get_scheduler
from .vitad.loss import get_loss_terms
from .vitad.util.net import cal_anomaly_map
from .base import BaseDetector


class HRVitAD(BaseDetector):

    def __init__(self,
                 patch_size,  # base
                 backbone_name,
                 config_path,
                 logger: logging.Logger,  #base
                 device: torch.device,  #base
                 seed = 0,  # base
                 fusion_weights = None,
                 early_stop_epochs=-1,
                 **kwargs):

        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)

        self.backbone_name = backbone_name
        self.config_path = config_path
        self.max_anomaly_score = None
        self.min_anomaly_score = None
        self.config_path = self.config_path.split('.')[0].replace('/', '.')
        dataset_lib = importlib.import_module(self.config_path)
        self.cfg = dataset_lib.VitADConfig(self.backbone_name, (self.patch_size[1], self.patch_size[0]))
        set_seed(self.seed)

        self.net = get_model(self.cfg.model)
        self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        input_tensor = input_tensor.to(self.device)
        feats_t, _ = self.net.encoder(input_tensor)
        return feats_t

    def create_dataset(self, patches: List[LRPatch], training: bool, task_name: str ):
        if not self.backbone_name.startswith('vit'):
            dataset = PatchDataset(patches=patches, training=training, task_name=task_name)
        else:
            if self.backbone_name.endswith('dino'):
                IMAGENET_MEAN = [0.485, 0.456, 0.406]
                IMAGENET_STD = [0.229, 0.224, 0.225]
            else:
                IMAGENET_MEAN = [0.5, 0.5, 0.5]
                IMAGENET_STD = [0.5, 0.5, 0.5]
            dataset = PatchDataset(patches=patches, training=training, task_name=task_name,
                                      normalize_mean = IMAGENET_MEAN,
                                    normalize_std = IMAGENET_STD)
        return dataset


    @torch.no_grad()
    def get_multi_resolution_feats_t_and_n(self, data):
        image = data['image'].to(self.device)
        feats_t, feats_n = self.net.encoder(image)
        low_resolution_image_keys = [key for key in data if key.startswith('low_resolution_image')]

        if len(low_resolution_image_keys) == 0:
            for i in range(len(feats_n)):
                feature = feats_n[i]
                B, C, H, W = feature.shape
                feats_n[i] = feature.permute(0, 2, 3, 1).view(B, H*W ,C).contiguous()
            return feats_t, feats_n
        else:
            if self.fusion_weights is not None:
                assert len(self.fusion_weights) == len(low_resolution_image_keys) + 1
                fusion_weights = [weight / sum(self.fusion_weights) for weight in self.fusion_weights]
            else:
                fusion_weights = [1 / (len(low_resolution_image_keys) + 1)] * (len(low_resolution_image_keys) + 1)

            feats_t_list = [[embedding * fusion_weights[0]] for embedding in feats_t]
            feats_n_list = [[embedding * fusion_weights[0]] for embedding in feats_n]

            low_resolution_image_keys.sort(key=lambda item: int(item.split('_')[-1]))

            for rs_index, low_resolution_image_key in enumerate(low_resolution_image_keys):
                low_resolution_images = data[low_resolution_image_key].to(self.device)
                low_resolution_indexes = data[low_resolution_image_key.replace('image', 'index')]
                low_resolution_images_feats_t, low_resolution_images_feats_n = self.net.encoder(low_resolution_images)

                for i, low_resolution_embedding in enumerate(low_resolution_images_feats_t):
                    downsampling_embedding = []
                    for feature, index in zip(low_resolution_embedding,  low_resolution_indexes):
                        stride_H, stride_W = self.patch_size[1] / feature.shape[1], self.patch_size[0] / feature.shape[2]
                        index = HRImageIndex.from_str(index)
                        x_start = index.x / stride_W
                        y_start = index.y / stride_H
                        x_end = x_start + index.width / stride_W
                        y_end = y_start + index.height / stride_H
                        downsampling_embedding.append(feature[:, int(y_start):int(y_end), int(x_start):int(x_end)])

                    try:
                        downsampling_embedding = torch.stack(downsampling_embedding)
                    except:
                        first_embedding = downsampling_embedding[0]
                        downsampling_embedding = [F.interpolate(feat.unsqueeze(0), size=(
                        first_embedding.shape[-2], first_embedding.shape[-1]), mode='bilinear').squeeze(0)
                                                  for feat in downsampling_embedding[1:]]
                        downsampling_embedding = [first_embedding] + downsampling_embedding
                        downsampling_embedding = torch.stack(downsampling_embedding)

                    feats_t_list[i].append(fusion_weights[rs_index+1] * F.interpolate(
                        downsampling_embedding,
                        size=(feats_t_list[i][-1].shape[-2], feats_t_list[i][-1].shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    ))

                for i, low_resolution_embedding in enumerate(low_resolution_images_feats_n):
                    downsampling_embedding = []
                    for feature, index in zip(low_resolution_embedding, low_resolution_indexes):
                        stride_H, stride_W = self.patch_size[1] / feature.shape[1], self.patch_size[0] / feature.shape[2]
                        index = HRImageIndex.from_str(index)
                        x_start = index.x / stride_W
                        y_start = index.y / stride_H
                        x_end = x_start + index.width / stride_W
                        y_end = y_start + index.height / stride_H
                        downsampling_embedding.append(feature[:, int(y_start):int(y_end), int(x_start):int(x_end)])
                    try:
                        downsampling_embedding = torch.stack(downsampling_embedding)
                    except:
                        first_embedding = downsampling_embedding[0]
                        downsampling_embedding = [F.interpolate(feat.unsqueeze(0), size=(
                            first_embedding.shape[-2], first_embedding.shape[-1]), mode='bilinear').squeeze(0)
                                                  for feat in downsampling_embedding[1:]]
                        downsampling_embedding = [first_embedding] + downsampling_embedding
                        downsampling_embedding = torch.stack(downsampling_embedding)

                    feats_n_list[i].append(fusion_weights[rs_index+1] * F.interpolate(
                        downsampling_embedding,
                        size=(feats_n_list[i][-1].shape[-2], feats_n_list[i][-1].shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    ))

            feats_t_list = [torch.sum(torch.stack(embedding), dim=0, keepdim=False) for embedding in feats_t_list]
            feats_n_list = [torch.sum(torch.stack(embedding), dim=0, keepdim=False) for embedding in feats_n_list]

            for i in range(len(feats_n_list)):
                feature = feats_n_list[i]
                B, C, H, W = feature.shape
                feats_n_list[i] = feature.permute(0, 2, 3, 1).view(B,H*W,C).contiguous()

            return feats_t_list, feats_n_list


    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators = None,
                   ) -> bool:

        optim = get_optim(self.cfg.optim.kwargs, self.net, lr=self.cfg.optim.lr)
        loss_terms = get_loss_terms(self.cfg.loss.loss_terms, device=self.device)

        self.cfg.train_size = len(train_dataloader)
        scheduler = get_scheduler(self.cfg, optim)

        train_iterator = iter(train_dataloader)

        best_metrics = {}
        epoch = 0
        iter_number = 0
        train_length = len(train_dataloader)

        while epoch < self.cfg.epoch_full:
            torch.cuda.empty_cache()
            self.net.train(mode=True)
            scheduler.step(iter_number)
            iter_number += 1
            train_data = next(train_iterator)
            feats_t_list, feats_n_list = self.get_multi_resolution_feats_t_and_n(train_data)
            feats_t, feats_s = self.net.decoder(feats_t_list, feats_n_list)
            loss_cos = loss_terms['cos'](feats_t, feats_s)
            optim.zero_grad()
            loss_cos.backward()

            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            optim.step()

            if iter_number % self.cfg.log_per_steps == 0:
                self.logger.info(
                    "Epoch {}/{} - Step {}/{}: loss = {:.3f}".format(
                        epoch + 1, self.cfg.epoch_full,
                        iter_number, self.cfg.epoch_full*train_length,
                        loss_cos.item()
                    )
                )

            if iter_number >= self.cfg.trainer.start_eval_steps and \
                    iter_number % self.cfg.trainer.eval_per_steps == 0 and \
                    val_dataloader is not None:

                best_metrics = self.val_step(val_dataloader, evaluators, checkpoint_path, best_metrics)
                if self.early_stop_cur == 0:
                    return True

            if iter_number % train_length == 0:
                epoch += 1
                train_iterator = iter(train_dataloader)

        return True if best_metrics!={} else False


    @torch.no_grad()
    def inference_step(self,
                   test_dataloader: DataLoader,
                   task_name: str,
                   )-> List[numpy.ndarray]:

        self.net.train(mode=False)
        anomaly_maps = []
        batch_idx = 0
        test_length = len(test_dataloader)
        test_loader = iter(test_dataloader)

        while batch_idx < test_length:
            batch_idx += 1
            test_data = next(test_loader)
            feats_t_list, feats_n_list = self.get_multi_resolution_feats_t_and_n(test_data)
            feats_t, feats_s = self.net.decoder(feats_t_list, feats_n_list)
            anomaly_map, _ = cal_anomaly_map(feats_t, feats_s,
                                             self.patch_size,
                                             uni_am=False, amap_mode='add', gaussian_sigma=4)
            anomaly_maps.append(anomaly_map)

        anomaly_maps = np.concatenate(anomaly_maps, axis=0)
        if task_name != 'val':
            anomaly_maps = self.patch_post_processing(anomaly_maps)
        return [pred for pred in anomaly_maps]


    def save_checkpoint(self, checkpoint_path: str):
        torch.save({'state_dict': self.net.state_dict(),
                    'max_anomaly_score': self.max_anomaly_score,
                    'min_anomaly_score': self.min_anomaly_score,
                    'fusion_weights': self.fusion_weights
                    }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(state_dict['state_dict'])
        self.min_anomaly_score = state_dict['min_anomaly_score']
        self.max_anomaly_score = state_dict['max_anomaly_score']
        self.fusion_weights = state_dict['fusion_weights'] if 'fusion_weights' in state_dict else None