from typing import List
import numpy
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from hiad.utils.split_and_gather import HRImageIndex, LRPatch
from hiad.datasets.patch_dataset import PatchDataset
from hiad.datasets.syn_dataset import AnomalySynDataset
from hiad.syn import *
from .base import BaseDetector
from .inpformer.models import vit_encoder
from .inpformer.dinov3.hub.backbones import load_dinov3_model
from .inpformer.models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from .inpformer.models.uad import INP_Former
from .inpformer.dinov1.utils import trunc_normal_
from .inpformer.optimizers import StableAdamW
from .inpformer.utils import WarmCosineScheduler, global_cosine_hm_adaptive


class HRINPFormer(BaseDetector):
    def __init__(self,
                 encoder_name,
                 total_iters,
                 eval_per_steps,
                 log_per_steps,
                 inp_num,
                 patch_size: int,  # base
                 logger: logging.Logger,  # base
                 device: torch.device,  # base
                 dinov3_pretrained_weight: str = None,
                 seed: int = 0,  #base
                 fusion_weights = None,
                 early_stop_epochs=-1,
                 **kwargs):

        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)

        self.encoder_name = encoder_name
        self.total_iters = total_iters
        self.dinov3_pretrained_weight = dinov3_pretrained_weight
        self.is_dinov3 = self.encoder_name.startswith('dinov3')
        self.inp_num = inp_num

        if self.is_dinov3:
            assert self.dinov3_pretrained_weight is not None, "If you use DINOv3, please make sure to set dinov3_pretrained_weight."

        self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        if 'vit_small' in self.encoder_name or 'vits' in self.encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'vit_base' in self.encoder_name or 'vitb' in self.encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'vit_large' in self.encoder_name or 'vitl' in self.encoder_name:
            embed_dim, num_heads = 1024, 16
            self.target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."

        if self.is_dinov3:
            self.encoder = load_dinov3_model(self.encoder_name,
                                            layers_to_extract_from = self.target_layers,
                                            pretrained_weight_path= self.dinov3_pretrained_weight)

            self.feat_map_size = (self.patch_size[1] // 16, self.patch_size[0] // 16)
        else:
            self.encoder = vit_encoder.load(self.encoder_name)
            self.feat_map_size = (self.patch_size[1] // 14, self.patch_size[0] // 14)

        self.Bottleneck = []
        self.Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
        self.Bottleneck = nn.ModuleList(self.Bottleneck)

        self.INP = nn.ParameterList(
            [nn.Parameter(torch.randn(self.inp_num, embed_dim))
             for _ in range(1)])

        self.INP_Extractor = []

        # INP Extractor
        for i in range(1):
            blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            self.INP_Extractor.append(blk)
        self.INP_Extractor = nn.ModuleList(self.INP_Extractor)

        self.INP_Guided_Decoder = []

        for i in range(8):
            blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            self.INP_Guided_Decoder.append(blk)
        self.INP_Guided_Decoder = nn.ModuleList(self.INP_Guided_Decoder)

        self.model = INP_Former(encoder=self.encoder, bottleneck=self.Bottleneck, aggregation=self.INP_Extractor,
                                decoder=self.INP_Guided_Decoder, is_dinov3=self.is_dinov3, feat_size=self.feat_map_size,
                                target_layers=self.target_layers, fuse_layer_encoder=self.fuse_layer_encoder,
                                fuse_layer_decoder=self.fuse_layer_decoder, prototype_token=self.INP)

        self.model = self.model.to(device)

        self.eval_per_steps = eval_per_steps
        self.log_per_steps = log_per_steps
        self.max_anomaly_score = None
        self.min_anomaly_score = None


    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        return self.model.encoder_image(input_tensor.to(self.device))


    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators = None,
                   ) -> bool:

        trainable = nn.ModuleList([self.Bottleneck, self.INP_Guided_Decoder, self.INP_Extractor, self.INP])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW([{'params': trainable.parameters()}], lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                                amsgrad = not self.is_dinov3, eps=1e-10)

        lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=self.total_iters, warmup_iters=100)

        it = 0
        best_metrics = {}

        for epoch in range(int(np.ceil(self.total_iters / len(train_dataloader)))):
            torch.cuda.empty_cache()

            for data in train_dataloader:

                self.model.train()
                self.model.encoder.eval()

                en = self.get_multi_resolution_fusion_embeddings(data)
                en, de, g_loss = self.model.distillation(en)

                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                lr_scheduler.step()
                it += 1

                if it % self.log_per_steps == 0:
                    self.logger.info('iter [{}/{}], loss:{:.4f}'.format(it, self.total_iters, loss.item()))

                if it % self.eval_per_steps == 0 and val_dataloader is not None:
                    best_metrics = self.val_step(val_dataloader, evaluators, checkpoint_path, best_metrics)
                    if self.early_stop_cur == 0:
                        return True

                if it == self.total_iters:
                    break

        return True if best_metrics != {} else False


    @torch.no_grad()
    def inference_step(self,
                   test_dataloader: DataLoader,
                   task_name: str,
                   )-> List[numpy.ndarray]:

        self.model.eval()
        preds = []
        for data in test_dataloader:

            en = self.get_multi_resolution_fusion_embeddings(data)
            en, de, _ = self.model.distillation(en)

            anomaly_map, _ = self.cal_anomaly_maps(en, de, self.patch_size)
            preds.append(anomaly_map.cpu().detach().numpy())

        preds = numpy.concatenate(preds, axis=0).squeeze(axis=1)
        if task_name != 'val':
            preds = self.patch_post_processing(preds)

        return [pred for pred in preds]


    def cal_anomaly_maps(self, fs_list, ft_list, out_size= (224, 224)):
        a_map_list = []
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=(out_size[1], out_size[0]), mode='bilinear', align_corners=True)
            a_map_list.append(a_map)
        anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
        return anomaly_map, a_map_list


    def save_checkpoint(self, checkpoint_path: str):
        torch.save({'Bottleneck': self.Bottleneck.state_dict(),
                    'INP_Guided_Decoder': self.INP_Guided_Decoder.state_dict(),
                    'INP_Extractor': self.INP_Extractor.state_dict(),
                    'INP': self.INP.state_dict(),
                    'max_anomaly_score': self.max_anomaly_score,
                    'min_anomaly_score': self.min_anomaly_score,
                    'fusion_weights': self.fusion_weights},
                   checkpoint_path)


    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.Bottleneck.load_state_dict(state_dict['Bottleneck'])
        self.INP_Extractor.load_state_dict(state_dict['INP_Extractor'])
        self.INP_Guided_Decoder.load_state_dict(state_dict['INP_Guided_Decoder'])
        self.INP.load_state_dict(state_dict['INP'])
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