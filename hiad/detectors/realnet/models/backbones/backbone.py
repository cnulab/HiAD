import torch
import timm


class Backbone(torch.nn.Module):
    def __init__(self,
                 backbone,
                 outlayers,
                 ):

        super(Backbone, self).__init__()
        self.backbone=backbone
        self.outlayers=outlayers

        assert self.backbone in ['resnet18','resnet34','resnet50','efficientnet_b4','wide_resnet50_2']

        if self.backbone =='resnet50' or self.backbone=='wide_resnet50_2':
            layers_idx ={'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4}
            layers_strides = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
            layers_planes= {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}

        elif self.backbone=='resnet34' or self.backbone=='resnet18':
            layers_idx = {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4': 4}
            layers_strides = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
            layers_planes = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}

        elif self.backbone == 'efficientnet_b4':
            # if you use efficientnet_b4 as backbone, make sure timm==0.5.x, we use 0.5.4
            layers_idx = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'layer5': 4}
            layers_strides = {'layer1': 2, 'layer2': 4, 'layer3': 8, 'layer4': 16, 'layer5': 32}
            layers_planes = {'layer1': 24, 'layer2': 32, 'layer3': 56, 'layer4': 160, 'layer5': 448}
        else:
            raise NotImplementedError("backbone must in [resnet18, resnet34,resnet50, wide_resnet50_2, efficientnet_b4]")

        self.feature_extractor = timm.create_model(self.backbone, features_only=True,pretrained=True,
                                                  out_indices=[layers_idx[outlayer] for outlayer in self.outlayers])
        self.layers_strides = layers_strides
        self.layers_planes = layers_planes


    @torch.no_grad()
    def forward(self, image, train=False):
        feats = self.feature_extractor(image)
        return feats


    def get_outplanes(self):
        return { outlayer: self.layers_planes[outlayer] for outlayer in self.outlayers}

    def get_outstrides(self):
        return { outlayer: self.layers_strides[outlayer] for outlayer in self.outlayers}
