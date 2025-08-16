from typing import List
import numpy
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from random import sample

from .base import BaseDetector
from .padim.utils import  fix_seeds
from .padim import backbones



class HRPaDiM(BaseDetector):

    def __init__(self,
                 patch_size: int, #base
                 logger: logging.Logger, #base
                 device: torch.device, #base
                 backbone_name: str,
                 layers_to_extract_from: List[str],
                 embed_dimension,
                 seed: int = 0, #base
                 fusion_weights = None,
                 early_stop_epochs=-1,
                 **kwargs):

        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)

        self.backbone_name = backbone_name
        self.layers_to_extract_from = layers_to_extract_from
        self.embed_dimension =  embed_dimension

        fix_seeds(seed)

        backbone = backbones.load(self.backbone_name).to(device)

        self.feature_aggregator = backbones.NetworkFeatureAggregator(
            backbone, self.layers_to_extract_from, self.device
        )
        self.feature_aggregator.eval()

        self.feature_dimension = sum(self.feature_dimensions())
        self.idx = torch.tensor(sample(range(0, self.feature_dimension), self.embed_dimension))

        self.max_anomaly_score = None
        self.min_anomaly_score = None


    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators = None,
                   ) -> bool:

        train_outputs = [[] for _ in  self.layers_to_extract_from]
        self.logger.info('Start Feature Embedding >...')
        for data in train_dataloader:

            with torch.no_grad():
                outputs = self.get_multi_resolution_fusion_embeddings(data)

            for train_output,output in zip(train_outputs, outputs):
                train_output.append(output.cpu().detach())

        train_outputs = [torch.cat(train_output, 0) for train_output in train_outputs]

        embedding_vectors = train_outputs[0]
        for train_output in train_outputs[1:]:
            embedding_vectors = self.embedding_concat(embedding_vectors, train_output)

        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        self.logger.info('Start Computing Mean and Cov >...')

        self.mean = torch.mean(embedding_vectors, dim=0)
        self.cov = torch.zeros(C, C, H * W)

        I = torch.from_numpy(np.identity(C)).to(self.device)

        for i in range(H * W):
            cov_ = torch.cov(embedding_vectors[:, :, i].T.to(self.device)) + 0.01 * I
            self.cov[:, :, i] = cov_.cpu()

        self.logger.info('Save Mean and Cov >...')
        self.logger.info('Computing Max and Min anomaly score >...')
        if val_dataloader is not None:
            pred_masks = self.inference_step(val_dataloader, task_name='val')
            self.max_anomaly_score, self.min_anomaly_score = np.max(np.stack(pred_masks)), np.min(np.stack(pred_masks))
        self.save_checkpoint(checkpoint_path)
        return True


    @torch.no_grad()
    def inference_step(self,
                   test_dataloader: DataLoader,
                   task_name: str,
                   )-> List[numpy.ndarray]:

        test_outputs = [[] for _ in  self.layers_to_extract_from]
        for data in test_dataloader:
            with torch.no_grad():
                outputs = self.get_multi_resolution_fusion_embeddings(data)

            for test_output,output in zip(test_outputs, outputs):
                test_output.append(output.cpu().detach())

        test_outputs = [torch.cat(test_output, 0) for test_output in test_outputs]

        embedding_vectors = test_outputs[0]
        for test_output in test_outputs[1:]:
            embedding_vectors = self.embedding_concat(embedding_vectors, test_output)

        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)

        dist_list = []
        for i in range(H * W):
            mean = self.mean[:, i].to(self.device)
            conv_inv = torch.linalg.inv(self.cov[:, :, i].to(self.device))
            dist = [self.mahalanobis(sample[:, i].to(self.device), mean, conv_inv).cpu().numpy() for sample in embedding_vectors]
            dist_list.append(dist)

        preds = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        preds = torch.tensor(preds)
        preds = F.interpolate(preds.unsqueeze(1), size=(self.patch_size[1], self.patch_size[0]), mode='bilinear',
                                      align_corners=False).squeeze(1).numpy()

        if task_name != 'val':
            preds = self.patch_post_processing(preds)
        return [pred for pred in preds]


    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        input_tensor = input_tensor.to(self.device)
        self.feature_aggregator.eval()
        outputs = self.feature_aggregator(input_tensor)
        outputs = [outputs[layer] for layer in self.layers_to_extract_from]
        if self.backbone_name.startswith('vit'):
            outputs_ = []
            for output in outputs:
                output = output[:,1:,:]
                B,L,C = output.shape
                outputs_.append(output.view((B,int(math.sqrt(L)),int(math.sqrt(L)),C)).permute(0, 3, 1, 2).contiguous())
            return outputs_
        else:
            return outputs


    def save_checkpoint(self, checkpoint_path: str):
        state_dict = {'mean': self.mean,
                      'cov': self.cov,
                      'idx': self.idx,
                      'max_anomaly_score': self.max_anomaly_score,
                      'min_anomaly_score':self.min_anomaly_score,
                      'fusion_weights': self.fusion_weights}
        torch.save(state_dict, checkpoint_path)


    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.mean = state_dict['mean']
        self.cov = state_dict['cov']
        self.idx = state_dict['idx'].cpu()
        self.min_anomaly_score = state_dict['min_anomaly_score']
        self.max_anomaly_score = state_dict['max_anomaly_score']
        self.fusion_weights = state_dict['fusion_weights'] if 'fusion_weights' in state_dict else None


    def embedding_concat(self, x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
        return z


    def mahalanobis(self, u, v, vi):
        diff = u - v
        diff_vi = torch.matmul(diff, vi)
        dist = torch.matmul(diff_vi, diff)
        return torch.sqrt(dist)


