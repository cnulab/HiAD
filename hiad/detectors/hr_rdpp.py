from typing import List
import numpy
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from hiad.utils.split_and_gather import LRPatch
from hiad.datasets.syn_dataset import AnomalySynDataset
from hiad.datasets.patch_dataset import PatchDataset
from hiad.syn import *
from .base import BaseDetector
from .rdpp import resnet
from .rdpp import de_resnet
from .rdpp.utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion


class HRRDPP(BaseDetector):

    def __init__(self,
                 patch_size: int, # base
                 backbone_name: str,
                 layers_to_extract_from: list,
                 synthesizer,
                 epoch: int,
                 proj_lr: float,
                 distill_lr: float,
                 weight_proj: float,
                 logger: logging.Logger, #base
                 device: torch.device, #base
                 seed: int = 0, #base
                 log_per_steps: int = 10,
                 eval_per_steps: int= 100,
                 fusion_weights = None,
                 early_stop_epochs = -1,
                 **kwargs):

        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)

        self.backbone_name = backbone_name
        self.epoch = epoch
        self.proj_lr = proj_lr
        self.distill_lr = distill_lr
        self.weight_proj =  weight_proj
        self.synthesizer = synthesizer
        self.log_per_steps = log_per_steps
        self.eval_per_steps = eval_per_steps

        self.layers_to_extract_from = layers_to_extract_from
        self.synthesizer = eval(self.synthesizer.name)(**self.synthesizer.kwargs)

        self.encoder, self.bn = resnet._BACKBONES[self.backbone_name](pretrained=True,
                                                                      layers_to_extract_from=self.layers_to_extract_from)
        self.encoder = self.encoder.to(device)
        self.bn = self.bn.to(device)
        self.encoder.eval()

        self.decoder = de_resnet._BACKBONES[self.backbone_name](pretrained=False, layers_to_extract_from=self.layers_to_extract_from)
        self.decoder = self.decoder.to(device)

        self.proj_layer = MultiProjectionLayer(base=64, layers_to_extract_from=self.layers_to_extract_from).to(device)

        self.max_anomaly_score = None
        self.min_anomaly_score = None

    def create_dataset(self, patches: List[LRPatch], training: bool, task_name: str ):
        if not training:
            dataset = PatchDataset(patches=patches, training=training, task_name=task_name)
        else:
            dataset = AnomalySynDataset(patches=patches, synthesizer=self.synthesizer, training=training, task_name = task_name)
        return dataset

    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor ) -> List[torch.Tensor]:
        input_tensor = input_tensor.to(self.device)
        outputs = self.encoder(input_tensor)
        return outputs

    def rdpp_emb(self, samples):
        normal_image = {'image': samples['gt_image']}
        noise_image =  {'image': samples['image']}
        for key in samples:
            if key.startswith('gt_low_resolution_index') or key.startswith('gt_low_resolution_image'):
                normal_image[key.replace('gt_', '')] = samples[key]

            if key.startswith('low_resolution_index') or key.startswith('low_resolution_image'):
                noise_image[key] = samples[key]

        features, noise_features = self.get_multi_resolution_fusion_embeddings(normal_image), self.get_multi_resolution_fusion_embeddings(noise_image)
        return features, noise_features


    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators = None,
                   ) -> bool:

        self.logger.info(f"RD++: {task_name} start training ...")
        proj_loss = Revisit_RDLoss()

        optimizer_proj = torch.optim.Adam(list(self.proj_layer.parameters()), lr=self.proj_lr, betas=(0.5, 0.999))
        optimizer_distill = torch.optim.Adam(list(self.decoder.parameters()) + list(self.bn.parameters()), lr=self.distill_lr,
                                             betas=(0.5, 0.999))

        best_metrics = {}
        global_step = 0

        for epoch in range(self.epoch):

            loss_list = []
            for data in train_dataloader:
                torch.cuda.empty_cache()
                self.bn.train()
                self.proj_layer.train()
                self.decoder.train()

                global_step += 1

                features, noise_features = self.rdpp_emb(data)

                (feature_space_noise, feature_space) = self.proj_layer(features, features_noise=noise_features)

                L_proj = proj_loss(noise_features, feature_space_noise, feature_space)
                outputs = self.decoder(self.bn(feature_space))  # bn(inputs))
                L_distill = loss_fucntion(features, outputs)

                loss = L_distill + self.weight_proj * L_proj
                loss.backward()

                if global_step % 2 == 0:
                    optimizer_proj.step()
                    optimizer_distill.step()
                    optimizer_proj.zero_grad()
                    optimizer_distill.zero_grad()

                loss_list.append(loss.item())

                if global_step % self.log_per_steps == 0 or global_step % len(train_dataloader) == 0:
                    self.logger.info('epoch [{}/{}], step [{}], loss:{:.4f}'.format(epoch + 1, self.epoch, global_step,
                                                                                    np.mean(loss_list)))

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

        self.bn.eval()
        self.decoder.eval()
        self.proj_layer.eval()

        preds = []
        with torch.no_grad():
            for data in test_dataloader:
                B = data['image'].shape[0]
                inputs = self.get_multi_resolution_fusion_embeddings(data)
                features = self.proj_layer(inputs)
                outputs = self.decoder(self.bn(features))

                inputs = [torch.split(input, 1) for input in inputs]
                outputs = [torch.split(output, 1) for output in outputs]

                for index in range(B):
                    input_ = [input[index] for input in inputs]
                    output_ = [output[index] for output in outputs]
                    anomaly_map, _ = self.cal_anomaly_map(input_, output_, self.patch_size, amap_mode='a')
                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                    preds.append(np.expand_dims(anomaly_map, axis=0))

        preds = np.concatenate(preds, axis=0)

        if task_name != 'val':
            preds = self.patch_post_processing(preds)

        preds = np.split(preds, indices_or_sections=preds.shape[0])
        preds = [pred.squeeze(axis=0) for pred in preds]
        return preds


    def save_checkpoint(self, checkpoint_path: str):
        torch.save({'bn': self.bn.state_dict(), 'decoder': self.decoder.state_dict(),
                    'proj_layer': self.proj_layer.state_dict(),
                    'max_anomaly_score': self.max_anomaly_score, 'min_anomaly_score': self.min_anomaly_score,
                    'fusion_weights': self.fusion_weights},
                    checkpoint_path)


    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.bn.load_state_dict(state_dict['bn'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.proj_layer.load_state_dict(state_dict['proj_layer'])
        self.min_anomaly_score = state_dict['min_anomaly_score']
        self.max_anomaly_score = state_dict['max_anomaly_score']
        self.fusion_weights = state_dict['fusion_weights'] if 'fusion_weights' in state_dict else None


    def cal_anomaly_map(self, fs_list, ft_list, out_size= (224, 224), amap_mode='mul'):
        if amap_mode == 'mul':
            anomaly_map = np.ones((out_size[1], out_size[0]))
        else:
            anomaly_map = np.zeros((out_size[1], out_size[0]))

        a_map_list = []
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]

            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=(out_size[1], out_size[0]), mode='bilinear', align_corners=True)
            a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
            a_map_list.append(a_map)
            if amap_mode == 'mul':
                anomaly_map *= a_map
            else:
                anomaly_map += a_map

        return anomaly_map, a_map_list
