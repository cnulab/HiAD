import torch
import torch.nn as nn
from torch.nn import functional as F
import geomloss
from typing import List

class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, in_c//4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//4),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//4, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, out_c, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )
    def forward(self, x):
        return self.proj(x)


class MultiProjectionLayer(nn.Module):
    def __init__(self, base, layers_to_extract_from):
        super(MultiProjectionLayer, self).__init__()
        self.projs = []
        if 'layer1' in layers_to_extract_from:
            self.projs.append(ProjLayer(base * 4, base * 4))
        if 'layer2' in layers_to_extract_from:
            self.projs.append(ProjLayer(base * 8, base * 8))
        if 'layer3' in layers_to_extract_from:
            self.projs.append(ProjLayer(base * 16, base * 16))

        for index, proj in enumerate(self.projs):
            self.add_module(f"proj_{index}", proj)

    def forward(self, features, features_noise: List = False):
        if features_noise is not False:
            return ( [ proj(feature) for feature, proj in zip(features_noise, self.projs)],
                    [ proj(feature) for feature, proj in zip(features, self.projs)])
        else:
            return [ proj(feature) for feature, proj in zip(features, self.projs)]


def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss


class CosineReconstruct(nn.Module):
    def __init__(self):
        super(CosineReconstruct, self).__init__()
    def forward(self, x, y):
        return torch.mean(1 - torch.nn.CosineSimilarity()(x, y))


class Revisit_RDLoss(nn.Module):
    """
    receive multiple inputs feature
    return multi-task loss:  SSOT loss, Reconstruct Loss, Contrast Loss
    """
    def __init__(self, consistent_shuffle = True):
        super(Revisit_RDLoss, self).__init__()
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05, \
                              reach=None, diameter=10000000, scaling=0.95, \
                                truncate=10, cost=None, kernel=None, cluster_scale=None, \
                                  debias=True, potentials=False, verbose=False, backend='tensorized')

        self.reconstruct = CosineReconstruct()       
        self.contrast = torch.nn.CosineEmbeddingLoss(margin = 0.5)

    def forward(self, noised_features, projected_noised_features, projected_normal_features):
        """
        noised_feature : output of encoder at each_blocks : [noised_feature_block1, noised_feature_block2, noised_feature_block3]
        projected_noised_feature: list of the projection layer's output on noised_features, projected_noised_feature = projection(noised_feature)
        projected_normal_feature: list of the projection layer's output on normal_features, projected_normal_feature = projection(normal_feature)
        """
        current_batchsize = projected_normal_features[0].shape[0]

        target = -torch.ones(current_batchsize).to('cuda')

        shuffle_index = torch.randperm(current_batchsize)

        shuffle_features  = [ feature[shuffle_index] for feature in projected_normal_features]

        loss_ssot = torch.sum(torch.stack([self.sinkhorn(torch.softmax(normal_proj.view(normal_proj.shape[0], -1), -1), torch.softmax(shuffle_feature.view(shuffle_feature.shape[0], -1), -1))
                               for normal_proj, shuffle_feature in zip(projected_normal_features, shuffle_features) ]))

        loss_reconstruct = torch.sum(torch.stack([self.reconstruct(abnormal_proj, normal_proj) for abnormal_proj, normal_proj in zip(projected_noised_features, projected_normal_features )]))

        loss_contrast = torch.sum(torch.stack([ self.contrast(noised_feature.view(noised_feature.shape[0], -1), normal_proj.view(normal_proj.shape[0], -1), target = target) for noised_feature, normal_proj in zip(noised_features, projected_normal_features )    ]))

        return (loss_ssot + 0.01 * loss_reconstruct + 0.1 * loss_contrast) / 1.11
