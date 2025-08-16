import logging
from typing import List
import torch
import numpy
from torch.utils.data import DataLoader
import torch.nn.functional as F

from hiad.datasets.patch_dataset import PatchDataset
from hiad.datasets.syn_dataset import AnomalySynDataset
from hiad.utils.split_and_gather import LRPatch,HRImageIndex
from hiad.syn import *
from .realnet.models.backbones import Backbone
from .realnet.models.model_helper import ModelHelper, to_device
from .realnet.utils.misc_helper import summary_model, AverageMeter
from .realnet.utils.optimizer_helper import get_optimizer
from .realnet.utils.criterion_helper import build_criterion
from .base import BaseDetector


class HRRealNet(BaseDetector):
    def __init__(self,
                 patch_size: int, # base
                 synthesizer,
                 structure,
                 net,
                 trainer,
                 criterion,
                 logger: logging.Logger, #base
                 device: torch.device, #base
                 seed: int = 0, #base
                 fusion_weights = None,
                 early_stop_epochs = -1,
                 **kwargs):
        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)

        self.synthesizer = synthesizer
        self.structure = structure
        self.net = net
        self.trainer = trainer
        self.criterion = criterion

        self.max_anomaly_score = None
        self.min_anomaly_score = None

        layers = []
        for block in self.structure:
            layers.extend([layer.idx for layer in block.layers])
        layers = list(set(layers))
        layers.sort()

        self.net[0].kwargs['outlayers'] = layers
        self.net[1].kwargs = self.net[1].get('kwargs', {})
        self.net[1].kwargs['structure'] = self.structure

        self.synthesizer = eval(self.synthesizer.name)(**self.synthesizer.kwargs)
        self.backbone = Backbone(**self.net[0].kwargs)
        self.backbone.to(self.device)
        self.backbone.eval()

        self.net = self.net[1:]
        if 'prev' in self.net[0]:
            del self.net[0]['prev']
            kwargs = self.net[0].get('kwargs',{})
            kwargs['inplanes'] = self.backbone.get_outplanes()
            kwargs['instrides'] = self.backbone.get_outstrides()
            self.net[0].kwargs = kwargs

        self.realnet = ModelHelper(self.net, self.device)
        self.realnet.to(self.device)


    def create_dataset(self, patches: List[LRPatch], training: bool, task_name: str ):
        if not training:
            dataset = PatchDataset(patches=patches, training=training, task_name=task_name)
        else:
            dataset = AnomalySynDataset(patches=patches, synthesizer=self.synthesizer, training=training, task_name = task_name)
        return dataset


    @torch.no_grad()
    def embedding(self, input_tensor, train = False):
        input_tensor = input_tensor.to(self.device)
        embeddings = self.backbone(input_tensor, train=train)
        return embeddings

    def realnet_emb(self, data, train = False):
        feats_dict = {}

        if train:
            feats, gt_feats = self.get_multi_resolution_fusion_embeddings(data, train)

            feats = {self.backbone.outlayers[idx]: {
                "feat": feats[idx],
                "stride": self.backbone.layers_strides[self.backbone.outlayers[idx]],
                "planes": self.backbone.layers_planes[self.backbone.outlayers[idx]],
            } for idx in range(len(self.backbone.outlayers))}

            feats_dict.update({"feats": feats})

            gt_feats = {self.backbone.outlayers[idx]: {
                "feat": gt_feats[idx],
                "stride": self.backbone.layers_strides[self.backbone.outlayers[idx]],
                "planes": self.backbone.layers_planes[self.backbone.outlayers[idx]],
            } for idx in range(len(self.backbone.outlayers))}
            feats_dict.update({"gt_feats": gt_feats})
        else:
            feats = self.get_multi_resolution_fusion_embeddings(data, train)

            feats = {self.backbone.outlayers[idx]: {
                "feat": feats[idx],
                "stride": self.backbone.layers_strides[self.backbone.outlayers[idx]],
                "planes": self.backbone.layers_planes[self.backbone.outlayers[idx]],
            } for idx in range(len(self.backbone.outlayers))}

            feats_dict.update({"feats": feats})
        return feats_dict


    @torch.no_grad()
    def get_multi_resolution_fusion_embeddings(self, data, train = False):
        data = to_device(data, device=self.device)
        image = data["image"]
        feats = self.embedding(image)
        low_resolution_image_keys = [key for key in data if key.startswith('low_resolution_image')]

        if len(low_resolution_image_keys) != 0:
            if self.fusion_weights is not None:
                assert len(self.fusion_weights) == len(low_resolution_image_keys) + 1
                fusion_weights = [weight / sum(self.fusion_weights) for weight in self.fusion_weights]
            else:
                fusion_weights = [1 / (len(low_resolution_image_keys) + 1)] * (len(low_resolution_image_keys) + 1)

            feats = [[feat * fusion_weights[0]] for feat in feats]
            low_resolution_image_keys.sort(key=lambda item: int(item.split('_')[-1]))
            for rs_index, low_resolution_image_key in enumerate(low_resolution_image_keys):
                low_resolution_images = data[low_resolution_image_key]
                low_resolution_indexes = data[low_resolution_image_key.replace('image', 'index')]
                low_resolution_embeddings = self.embedding(low_resolution_images)
                for i, low_resolution_embedding in enumerate(low_resolution_embeddings):
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
                        downsampling_embedding = [
                            F.interpolate(feat.unsqueeze(0), size=(first_embedding.shape[-2], first_embedding.shape[-1]),
                                          mode='bilinear').squeeze(0)
                            for feat in downsampling_embedding[1:]]
                        downsampling_embedding = [first_embedding] + downsampling_embedding
                        downsampling_embedding = torch.stack(downsampling_embedding)

                    feats[i].append(fusion_weights[rs_index+1] * F.interpolate(
                        downsampling_embedding,
                        size=(feats[i][-1].shape[-2], feats[i][-1].shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    ))

            feats = [torch.sum(torch.stack(feat), dim=0, keepdim=False) for feat in feats]

        if train:
            gt_image = data["gt_image"]
            gt_feats = self.embedding(gt_image)
            gt_low_resolution_image_keys = [key for key in data if key.startswith('gt_low_resolution_image')]

            if len(gt_low_resolution_image_keys) != 0:
                if self.fusion_weights is not None:
                    assert len(self.fusion_weights) == len(low_resolution_image_keys) + 1
                    fusion_weights = [weight / sum(self.fusion_weights) for weight in self.fusion_weights]
                else:
                    fusion_weights = [1 / (len(low_resolution_image_keys) + 1)] * (len(low_resolution_image_keys) + 1)

                gt_feats = [[feat * fusion_weights[0]] for feat in gt_feats]
                gt_low_resolution_image_keys.sort(key=lambda item: int(item.split('_')[-1]))

                for rs_index, gt_low_resolution_image_key in enumerate(gt_low_resolution_image_keys):
                    gt_low_resolution_images = data[gt_low_resolution_image_key]
                    gt_low_resolution_indexes = data[gt_low_resolution_image_key.replace('image', 'index')]
                    gt_low_resolution_embeddings = self.embedding(gt_low_resolution_images)

                    for i, gt_low_resolution_embedding in enumerate(gt_low_resolution_embeddings):
                        downsampling_embedding = []
                        for feature, index in zip(gt_low_resolution_embedding, gt_low_resolution_indexes):
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
                            downsampling_embedding = [
                                F.interpolate(feat.unsqueeze(0),
                                              size=(first_embedding.shape[-2], first_embedding.shape[-1]),
                                              mode='bilinear').squeeze(0)
                                for feat in downsampling_embedding[1:]]
                            downsampling_embedding = [first_embedding] + downsampling_embedding
                            downsampling_embedding = torch.stack(downsampling_embedding)

                        gt_feats[i].append(fusion_weights[rs_index+1] * F.interpolate(
                            downsampling_embedding,
                            size=(gt_feats[i][-1].shape[-2], gt_feats[i][-1].shape[-1]),
                            mode="bilinear",
                            align_corners=False,
                        ))
                gt_feats = [torch.sum(torch.stack(feat), dim=0, keepdim=False) for feat in gt_feats]

            return feats, gt_feats

        return feats


    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators = None,
                   ) -> bool:

        self.realnet.afs.init_idxs(self, train_dataloader)

        summary_model(self.realnet, self.logger)
        layers = []

        for module in self.net:
            layers.append(module["name"])

        frozen_layers = self.realnet.frozen_layers
        active_layers = [layer for layer in layers if layer not in frozen_layers]

        self.logger.info("layers: {}".format(layers))
        self.logger.info("frozen layers: {}".format(frozen_layers))
        self.logger.info("active layers: {}".format(active_layers))

        parameters = [
            {"params": getattr(self.realnet, layer).parameters()} for layer in active_layers
        ]
        optimizer = get_optimizer(parameters, self.trainer.optimizer)

        best_metrics = {}
        last_epoch = 0

        criterion = build_criterion(self.criterion)

        for epoch in range(last_epoch, self.trainer.max_epoch):

            last_iter = epoch * len(train_dataloader)
            losses = AverageMeter(self.trainer.log_per_steps)

            for i, input in enumerate(train_dataloader):
                torch.cuda.empty_cache()
                self.realnet.train()
                curr_step = last_iter + i + 1

                outputs = self.realnet_emb(input, train=True)
                outputs.update(input)
                outputs = self.realnet(outputs, train =True)

                loss = []
                for name, criterion_loss in criterion.items():
                    weight = criterion_loss.weight
                    loss.append(weight * criterion_loss(outputs))

                loss = torch.sum(torch.stack(loss))
                losses.update(loss.item())
                optimizer.zero_grad()
                loss.backward()
                if self.trainer.get("clip_max_norm", None):
                    max_norm = self.trainer.clip_max_norm
                    torch.nn.utils.clip_grad_norm_(self.realnet.parameters(), max_norm)
                optimizer.step()

                if curr_step % self.trainer.log_per_steps == 0:
                    self.logger.info(
                            "Epoch: [{0}/{1}]\t"
                            "Iter: [{2}/{3}]\t"
                            "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                                .format(
                                epoch + 1,
                                self.trainer.max_epoch,
                                curr_step,
                                len(train_dataloader) * self.trainer.max_epoch,
                                loss=losses,
                        )
                    )

                if curr_step % self.trainer.eval_per_steps == 0 and val_dataloader is not None:
                    best_metrics = self.val_step(val_dataloader, evaluators, checkpoint_path, best_metrics)
                    if self.early_stop_cur == 0:
                        return True

        return True if best_metrics!={} else False


    @torch.no_grad()
    def inference_step(self,
                   test_dataloader: DataLoader,
                   task_name: str,
                   )-> List[numpy.ndarray]:

        self.realnet.eval()
        self.backbone.eval()

        preds = []
        with torch.no_grad():
            for i, input in enumerate(test_dataloader):
                outputs = self.realnet_emb(input, train=False)
                outputs.update(input)
                outputs = self.realnet(outputs, train = False)
                preds.append(outputs["anomaly_score"])
        preds = torch.cat(preds, dim=0).cpu().numpy()
        assert preds.shape[0] == len(test_dataloader.dataset)
        preds = numpy.squeeze(preds, axis=1)
        if task_name != 'val':
            preds = self.patch_post_processing(preds)
        preds = [pred for pred in preds]  # N x H x W
        return preds

    def save_checkpoint(self, checkpoint_path: str):
        torch.save({'state_dict': self.realnet.state_dict(),
                    'max_anomaly_score': self.max_anomaly_score,
                    'min_anomaly_score': self.min_anomaly_score,
                    'fusion_weights': self.fusion_weights,
        }, checkpoint_path)


    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.realnet.load_state_dict(state_dict['state_dict'])
        self.min_anomaly_score = state_dict['min_anomaly_score']
        self.max_anomaly_score = state_dict['max_anomaly_score']
        self.fusion_weights = state_dict['fusion_weights'] if 'fusion_weights' in state_dict else None

