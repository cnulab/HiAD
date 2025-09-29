import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            is_dinov3,
            feat_size,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.is_dinov3 = is_dinov3
        self.feat_size = feat_size
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder

    def encoder_image(self, x):
        if self.is_dinov3:
            en_list = self.encoder.get_intermediate_layers(x, n=self.target_layers, norm=False)
        else:
            x = self.encoder.prepare_tokens(x)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.target_layers[-1]:
                    with torch.no_grad():
                        x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en_list.append(x[:, 1 + self.encoder.num_register_tokens:, :])

        feats = []
        for en in en_list:
            B, L, C = en.shape
            en = en.permute(0, 2, 1).reshape((B, C, self.feat_size[0], self.feat_size[1]))
            feats.append(en)
        return feats


    def distillation(self, en_feats):

        for idx, en in enumerate(en_feats):
            B, C, H, W = en.shape
            en_feats[idx] = en.reshape((B, C, H*W)).permute(0, 2, 1).contiguous()

        x = self.fuse_feature(en_feats)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_feats[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        for idx, e in enumerate(en):
            en[idx] = e.permute(0, 2, 1).reshape([x.shape[0], -1, self.feat_size[0], self.feat_size[1]]).contiguous()

        for idx, d in enumerate(de):
            de[idx] = d.permute(0, 2, 1).reshape([x.shape[0], -1,  self.feat_size[0], self.feat_size[1]]).contiguous()

        return en, de


    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)
