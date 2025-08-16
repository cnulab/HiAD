from typing import List, Union
import numpy
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from .patchcore import sampler

try:
    from .patchcore import common
except:
    print('faiss is not installed yet. PatchCore will be unavailable.')

from .patchcore import backbones
from .patchcore.utils import fix_seeds
from .base import BaseDetector


class HRPatchCore(BaseDetector):

    def __init__(self,
                 patch_size: Union[int,List], #base
                 logger: logging.Logger, #base
                 device: torch.device, #base
                 backbone_name: str,
                 layers_to_extract_from: List[str],
                 merge_size: int,
                 percentage: float,
                 pretrain_embed_dimension:int = 1024,
                 target_embed_dimension:int = 2048,
                 preprocessing: str ='mean',
                 aggregation: str = 'mean',
                 patch_score: str = 'max',
                 patch_overlap: float = 0.0,
                 anomaly_scorer_num_nn: int = 5,
                 faiss_on_gpu: bool = True,
                 faiss_num_workers: int = 8,
                 seed: int = 0, #base
                 fusion_weights = None,
                 early_stop_epochs=-1,
                 **kwargs):

        super().__init__(patch_size, device, fusion_weights, logger, seed, early_stop_epochs)

        self.backbone_name = backbone_name
        self.layers_to_extract_from = layers_to_extract_from
        self.merge_size = merge_size
        self.pretrain_embed_dimension = pretrain_embed_dimension
        self.target_embed_dimension = target_embed_dimension
        self.preprocessing = preprocessing
        self.aggregation = aggregation
        self.patch_score = patch_score
        self.percentage = percentage
        self.patch_overlap = patch_overlap
        self.anomaly_scorer_num_nn = anomaly_scorer_num_nn
        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers

        fix_seeds(seed)
        backbone = backbones.load(self.backbone_name)
        self.backbone = backbone.to(device)

        self.featuresampler = self.create_sampler('approx_greedy_coreset', self.percentage, self.logger)(
            self.device,
        )

        self.patch_maker = common.PatchMaker(self.merge_size, stride= 1)

        self.forward_modules = torch.nn.ModuleDict({})

        self.feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )

        self.nn_method = common.FaissNN(self.device, self.faiss_on_gpu, self.faiss_num_workers)

        self.forward_modules["feature_aggregator"] = self.feature_aggregator
        self.forward_modules["feature_aggregator"].eval()

        feature_dimensions = self.feature_dimensions()
        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        self.anomaly_scorer = common.NearestNeighbourScorer(
                    n_nearest_neighbours = self.anomaly_scorer_num_nn, nn_method=self.nn_method
        )
        self.anomaly_segmentor = common.RescaleSegmentor(
                    device=self.device, target_size = self.patch_size
            )
        self.max_anomaly_score = None
        self.min_anomaly_score = None


    def create_sampler(self, name, percentage ,logger):
        def get_sampler(device):
            if name == "identity":
                return sampler.IdentitySampler()
            elif name == "greedy_coreset":
                return sampler.GreedyCoresetSampler(percentage, device, logger)
            elif name == "approx_greedy_coreset":
                return sampler.ApproximateGreedyCoresetSampler(percentage, device, logger)
        return  get_sampler


    def patchcore_emb(self, samples, detach = True, provide_patch_shapes = False):

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        self.forward_modules["feature_aggregator"].eval()

        with torch.no_grad():
            features = self.get_multi_resolution_fusion_embeddings(samples)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]

        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)


    def train_step(self,
                   train_dataloader: DataLoader,
                   task_name: str,
                   checkpoint_path: str,
                   val_dataloader: DataLoader = None,
                   evaluators = None,
                   ) -> bool:

        self.forward_modules.eval()
        features = []
        for samples in train_dataloader:
            features.append(self.patchcore_emb(samples))
        features = np.concatenate(features, axis=0)
        self.patch_features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[self.patch_features])
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

        self.forward_modules.eval()
        pred_masks = []
        for data in test_dataloader:
            with torch.no_grad():
                batchsize = data['image'].shape[0]
                features, patch_shapes = self.patchcore_emb(data, provide_patch_shapes=True)
                features = np.asarray(features)

                patch_scores = self.anomaly_scorer.predict([features])[0]
                patch_scores = self.patch_maker.unpatch_scores(
                            patch_scores, batchsize=batchsize
                        )
                scales = patch_shapes[0]
                patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
                masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            pred_masks.extend([mask for mask in masks])

        if task_name != 'val':
            pred_masks = self.patch_post_processing(np.stack(pred_masks))
            pred_masks = [mask for mask in pred_masks]

        return pred_masks


    @torch.no_grad()
    def embedding(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        input_tensor = input_tensor.to(self.device)
        self.forward_modules["feature_aggregator"].eval()
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
        assert self.patch_features is not None
        state_dict = {'patch_features': self.patch_features,
                      'max_anomaly_score': self.max_anomaly_score,
                      'min_anomaly_score': self.min_anomaly_score,
                      'fusion_weights': self.fusion_weights}
        torch.save(state_dict, checkpoint_path)


    def load_checkpoint(self, checkpoint_path: str):
        assert self.anomaly_scorer is not None
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.anomaly_scorer.fit(detection_features=[state_dict['patch_features']])
        self.min_anomaly_score = state_dict['min_anomaly_score']
        self.max_anomaly_score = state_dict['max_anomaly_score']
        self.fusion_weights = state_dict['fusion_weights'] if 'fusion_weights' in state_dict else None
