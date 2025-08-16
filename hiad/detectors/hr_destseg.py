from typing import List
import numpy
import numpy as np
import logging
from torch.utils.data import DataLoader

from hiad.utils.split_and_gather import HRImageIndex, LRPatch
from hiad.datasets.patch_dataset import PatchDataset
from hiad.datasets.syn_dataset import AnomalySynDataset
from hiad.syn import *
from .base import BaseDetector
from .destseg.model.destseg import TeacherNet, StudentNet, SegmentationNet
from .destseg.model.model_utils import l2_normalize
from .destseg.model.losses import *


class HRDesTSeg(BaseDetector):

    def __init__(self,
                 patch_size: int,  # base
                 DTD,
                 steps,
                 de_st_steps,
                 logger: logging.Logger,  # base
                 device: torch.device,  # base
                 dest = True,
                 lr_de_st = 0.4,
                 lr_res = 0.1,
                 lr_seghead = 0.01,
                 eval_per_steps = 500,
                 log_per_steps = 50,
                 gamma = 4,
                 seed: int = 0,  #base
                 fusion_weights = None,
                 early_stop_epochs=-1,
                 **kwargs):

        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)

        self.max_anomaly_score = None
        self.min_anomaly_score = None
        self.steps = steps
        self.de_st_steps =  de_st_steps

        self.lr_de_st = lr_de_st
        self.lr_res = lr_res
        self.lr_seghead = lr_seghead

        self.eval_per_steps = eval_per_steps
        self.log_per_steps = log_per_steps
        self.gamma = gamma
        self.DTD = DTD
        self.dest = dest
        self.teacher_net = TeacherNet().to(self.device)
        self.student_net = StudentNet().to(self.device)
        self.segmentation_net = SegmentationNet(inplanes = 384).to(self.device)


    def create_dataset(self, patches: List[LRPatch], training: bool, task_name: str ):
        if not training:
            dataset = PatchDataset(patches=patches, training=training, task_name=task_name)
        else:
            dataset = AnomalySynDataset(patches=patches, synthesizer = ImageBlendingSynthesizer(p=1.0, anomaly_source = self.DTD),
                                        training=training, task_name = task_name)
        return dataset

    def destseg_embedding(self, input_tensor: torch.Tensor, model):
        return model(input_tensor)

    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        return self.destseg_embedding(input_tensor, self.teacher_net)

    @torch.no_grad()
    def get_multi_resolution_fusion_embeddings(self, data) -> List[torch.Tensor]:
        return self.destseg_multi_resolution_fusion_embeddings(data, self.teacher_net, gt_image=False)

    def destseg_multi_resolution_fusion_embeddings(self, data, model, gt_image = False) -> List[torch.Tensor]:
        image = data['gt_image'].to(self.device) if gt_image else data['image'].to(self.device)
        embeddings = self.destseg_embedding(image, model)
        low_resolution_image_keys = [key for key in data if key.startswith('gt_low_resolution_image' if gt_image else 'low_resolution_image')]
        if len(low_resolution_image_keys) == 0:
            return embeddings
        else:
            if self.fusion_weights is not None:
                assert len(self.fusion_weights) == len(low_resolution_image_keys) + 1, "fusion_weights must be the same length as ds_factor."
                fusion_weights = [weight / sum(self.fusion_weights) for weight in self.fusion_weights]
            else:
                fusion_weights = [1 / (len(low_resolution_image_keys) + 1)] * (len(low_resolution_image_keys) + 1)

            embeddings = [[embedding * fusion_weights[0]] for embedding in embeddings]

            low_resolution_image_keys.sort(key=lambda item: int(item.split('_')[-1]))

            for rs_index, low_resolution_image_key in enumerate(low_resolution_image_keys):
                low_resolution_images = data[low_resolution_image_key].to(self.device)
                low_resolution_indexes = data[low_resolution_image_key.replace('image', 'index')]
                low_resolution_embeddings = self.destseg_embedding(low_resolution_images, model)

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

                    embeddings[i].append(fusion_weights[rs_index+1] * F.interpolate(
                        downsampling_embedding,
                        size=(embeddings[i][-1].shape[-2], embeddings[i][-1].shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    ))
            embeddings = [torch.sum(torch.stack(embedding), dim=0, keepdim=False) for embedding in embeddings]
            return embeddings


    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators = None,
                   ) -> bool:

        seg_optimizer = torch.optim.SGD(
            [
                {"params": self.segmentation_net.res.parameters(), "lr": self.lr_res},
                {"params": self.segmentation_net.head.parameters(), "lr": self.lr_seghead},
            ],
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
        )

        de_st_optimizer = torch.optim.SGD(
            [
                {"params": self.student_net.parameters(), "lr": self.lr_de_st},
            ],
            lr=0.4,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
        )
        global_step = 0
        best_metrics = {}

        while global_step <= self.steps:
            torch.cuda.empty_cache()
            for _, sample_batched in enumerate(train_dataloader):
                seg_optimizer.zero_grad()
                de_st_optimizer.zero_grad()

                if global_step < self.de_st_steps:
                    self.student_net.train()
                    self.segmentation_net.eval()
                else:
                    self.student_net.eval()
                    self.segmentation_net.train()

                outputs_teacher_aug = [
                    l2_normalize(output_t.detach()) for output_t in self.destseg_multi_resolution_fusion_embeddings(sample_batched,
                                                                                                          self.teacher_net,
                                                                                                          gt_image = False)
                ]

                outputs_student_aug = [
                    l2_normalize(output_s) for output_s in self.destseg_multi_resolution_fusion_embeddings(sample_batched,
                                                                                                 self.student_net,
                                                                                                gt_image =  False)
                ]

                output = torch.cat(
                    [
                        F.interpolate(
                            -output_t * output_s,
                            size=outputs_student_aug[0].size()[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
                    ],
                    dim=1,
                )

                output_segmentation = self.segmentation_net(output)

                if self.dest:
                    outputs_student = outputs_student_aug
                else:
                    outputs_student = [
                        l2_normalize(output_s) for output_s in self.destseg_multi_resolution_fusion_embeddings(sample_batched, self.student_net,
                                                                                                     gt_image=True)
                    ]

                outputs_teacher = [
                    l2_normalize(output_t.detach()) for output_t in self.destseg_multi_resolution_fusion_embeddings(sample_batched, self.teacher_net,
                                                                                                     gt_image = True)
                ]

                output_de_st_list = []
                for output_t, output_s in zip(outputs_teacher, outputs_student):
                    a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
                    output_de_st_list.append(a_map)

                mask = F.interpolate(
                    sample_batched['mask'].unsqueeze(1),
                    size=output_segmentation.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                ).to(self.device)

                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )
                cosine_loss_val = cosine_similarity_loss(output_de_st_list)
                focal_loss_val = focal_loss(output_segmentation, mask, gamma=self.gamma)
                l1_loss_val = l1_loss(output_segmentation, mask)

                if global_step < self.de_st_steps:
                    total_loss_val = cosine_loss_val
                    total_loss_val.backward()
                    de_st_optimizer.step()
                else:
                    total_loss_val = focal_loss_val + l1_loss_val
                    total_loss_val.backward()
                    seg_optimizer.step()

                global_step += 1
                if global_step % self.log_per_steps == 0:
                    if global_step < self.de_st_steps:
                        self.logger.info(
                            f"Training at global step {global_step}/{self.steps}, cosine loss: {round(float(cosine_loss_val), 4)}"
                        )
                    else:
                        self.logger.info(
                            f"Training at global step {global_step}/{self.steps}, focal loss: {round(float(focal_loss_val), 4)}, l1 loss: {round(float(l1_loss_val), 4)}"
                        )

                if global_step >= self.de_st_steps and global_step % self.eval_per_steps == 0 and val_dataloader is not None:
                    best_metrics = self.val_step(val_dataloader, evaluators, checkpoint_path, best_metrics)
                    if self.early_stop_cur == 0:
                        return True

        return True if best_metrics!={} else False


    @torch.no_grad()
    def inference_step(self,
                   test_dataloader: DataLoader,
                   task_name: str,
                   )-> List[numpy.ndarray]:

        self.teacher_net.eval()
        self.student_net.eval()
        self.segmentation_net.eval()

        preds = []
        with torch.no_grad():
            for _, sample_batched in enumerate(test_dataloader):

                outputs_teacher_aug = [
                    l2_normalize(output_t.detach()) for output_t in self.destseg_multi_resolution_fusion_embeddings(sample_batched,
                                                                                                          self.teacher_net,
                                                                                                          gt_image=False)
                ]

                outputs_student_aug = [
                    l2_normalize(output_s) for output_s in self.destseg_multi_resolution_fusion_embeddings(sample_batched,
                                                                                                 self.student_net,
                                                                                                 gt_image=False)
                ]

                output = torch.cat(
                    [
                        F.interpolate(
                            -output_t * output_s,
                            size=outputs_student_aug[0].size()[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
                    ],
                    dim=1,
                )

                output_segmentation = self.segmentation_net(output)

                preds.append(F.interpolate(
                    output_segmentation,
                    size = sample_batched['mask'].size()[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        if task_name != 'val':
            preds = self.patch_post_processing(preds)
        preds = [pred for pred in preds]
        return preds


    def save_checkpoint(self, checkpoint_path: str):
        torch.save({'teacher': self.teacher_net.state_dict(),
                    'student': self.student_net.state_dict(),
                    'segmentation': self.segmentation_net.state_dict(),
                    'max_anomaly_score': self.max_anomaly_score,
                    'min_anomaly_score': self.min_anomaly_score,
                    'fusion_weights': self.fusion_weights},
                   checkpoint_path)


    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.teacher_net.load_state_dict(state_dict['teacher'])
        self.student_net.load_state_dict(state_dict['student'])
        self.segmentation_net.load_state_dict(state_dict['segmentation'])
        self.min_anomaly_score = state_dict['min_anomaly_score']
        self.max_anomaly_score = state_dict['max_anomaly_score']
        self.fusion_weights = state_dict['fusion_weights'] if 'fusion_weights' in state_dict else None


    @staticmethod
    def get_image_score(segmentations, T = 100, batch_size = 32):
        preds = torch.tensor(segmentations)
        preds = preds.split(batch_size, dim=0)
        image_scores = []
        for pred in preds:
            image_score, _ = torch.sort(
                pred.view(pred.size(0), -1).cuda(),
                dim=1,
                descending=True,
            )
            image_score = torch.mean(
                image_score[:, : T], dim=1
            )
            image_scores.append(image_score.cpu().numpy())

        image_scores = np.concatenate(image_scores, axis=0)
        return image_scores