from typing import List
import numpy
import torch
from torch.utils.data import DataLoader
import timm
import logging
import math

from hiad.datasets.patch_dataset import PatchDataset
from hiad.utils.split_and_gather import LRPatch
from .base import BaseDetector
from .fastflow import constants, fastflow


class HRFastFlow(BaseDetector):

    def __init__(self,
                 patch_size: int,  # base
                 backbone_name,
                 num_epochs,
                 logger: logging.Logger,  #base
                 device: torch.device,  #base
                 layers_to_extract_from = None,
                 seed = 0,  # base
                 flow_step = 8,
                 hidden_ratio = 0.1,
                 conv3x3_only = False,
                 lr = 1e-3,
                 weight_decay = 1e-5,
                 eval_per_steps = 100,
                 log_per_steps = 10,
                 fusion_weights = None,
                 early_stop_epochs = -1,
                 **kwargs):

        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)

        self.backbone_name = backbone_name
        self.layers_to_extract_from = layers_to_extract_from
        self.flow_step = flow_step
        self.hidden_ratio = hidden_ratio
        self.conv3x3_only = conv3x3_only
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.eval_per_steps = eval_per_steps
        self.log_per_steps = log_per_steps

        assert self.backbone_name in constants.SUPPORTED_BACKBONES

        if self.backbone_name in [constants.BACKBONE_RESNET18, constants.BACKBONE_WIDE_RESNET50]:
            assert self.layers_to_extract_from is not None

        if self.backbone_name in [constants.BACKBONE_RESNET18, constants.BACKBONE_WIDE_RESNET50]:
            layers_idx = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices= [layers_idx[layer] for layer in self.layers_to_extract_from],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()
        else:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [constants.SCALE_MAP[self.backbone_name]]

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)

        self.fastflow = fastflow.FastFlow(
            backbone_name=self.backbone_name,
            input_size=self.patch_size,
            flow_steps=self.flow_step,
            channels=channels,
            scales=scales,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio)

        self.fastflow.to(self.device)

        self.max_anomaly_score = None
        self.min_anomaly_score = None

    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        self.feature_extractor.eval()
        if isinstance(
                self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(input_tensor)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)

            if not hasattr(self.feature_extractor, 'dist_token') or self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )

            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)

            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)

            if not hasattr(self.feature_extractor, 'dist_token') or self.feature_extractor.dist_token is None:
                x = x[:, 1:, :]
            else:
                x = x[:, 2:, :]

            N, L, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, int(math.sqrt(L)), int(math.sqrt(L)))
            features = [x]

        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(input_tensor)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, L, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, int(math.sqrt(L)), int(math.sqrt(L)))
            features = [x]

        else:
            features = self.feature_extractor(input_tensor)
        return features

    def create_dataset(self, patches: List[LRPatch], training: bool, task_name: str):
        if self.backbone_name.startswith('vit'):
            IMAGENET_MEAN = [0.5, 0.5, 0.5]
            IMAGENET_STD = [0.5, 0.5, 0.5]
            dataset = PatchDataset(patches=patches, training=training, task_name=task_name,
                                      normalize_mean=IMAGENET_MEAN,
                                      normalize_std=IMAGENET_STD)
        else:
            dataset = PatchDataset(patches=patches, training=training, task_name=task_name)

        return dataset


    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators = None,
                   ) -> bool:

        optimizer = torch.optim.Adam(self.fastflow.parameters(), lr = float(self.lr), weight_decay = float(self.weight_decay))
        best_metrics = {}
        global_step = 0
        for epoch in range(self.num_epochs):
            loss_meter = fastflow.AverageMeter()
            for step, data in enumerate(train_dataloader):
                torch.cuda.empty_cache()
                self.fastflow.train()
                global_step += 1
                features = self.get_multi_resolution_fusion_embeddings(data)
                ret = self.fastflow(features)
                loss = ret["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item())
                if global_step % self.log_per_steps == 0 or global_step % len(train_dataloader) == 0:
                    self.logger.info(
                        "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                            epoch + 1, global_step, loss_meter.val, loss_meter.avg
                        )
                    )

                if global_step % self.eval_per_steps == 0 and val_dataloader is not None:
                    best_metrics = self.val_step(val_dataloader, evaluators, checkpoint_path, best_metrics)
                    if self.early_stop_cur == 0:
                        return True

        return True if best_metrics != {} else False


    @torch.no_grad()
    def inference_step(self,
                   test_dataloader: DataLoader,
                   task_name: str,
                   )-> List[numpy.ndarray]:

        self.feature_extractor.eval()
        self.fastflow.eval()
        preds = []
        for data in test_dataloader:
            features = self.get_multi_resolution_fusion_embeddings(data)
            ret = self.fastflow(features)
            preds.append(ret["anomaly_map"].cpu().detach().numpy())
        preds = numpy.concatenate(preds, axis=0).squeeze(axis=1)
        if task_name != 'val':
            preds = self.patch_post_processing(preds)
        return [pred for pred in preds]

    def save_checkpoint(self, checkpoint_path: str):
        torch.save({'state_dict': self.fastflow.state_dict(),
                    'max_anomaly_score': self.max_anomaly_score,
                    'min_anomaly_score': self.min_anomaly_score,
                    'fusion_weights': self.fusion_weights
                    }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.fastflow.load_state_dict(state_dict['state_dict'])
        self.min_anomaly_score = state_dict['min_anomaly_score']
        self.max_anomaly_score = state_dict['max_anomaly_score']
        self.fusion_weights = state_dict['fusion_weights'] if 'fusion_weights' in state_dict else None
