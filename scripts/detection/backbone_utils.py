from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter

import detnet
import resnet

"""
BSD 3-Clause License
Copyright (c) Soumith Chintala 2016, 
All rights reserved.
"""

#from torchvision.models import resnet


class BackboneWithFPN(nn.Sequential):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        super(BackboneWithFPN, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))
        self.out_channels = out_channels


def resnet_fpn_backbone(backbone_name, pretrained, n_channel_backbone = 5, first_layer_out = 'layer1'):

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # freeze layers

    in_channels_stage2 = backbone.inplanes // 8

    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]

    if n_channel_backbone == 1 :
        for name, parameter in backbone.named_parameters():
            if 'layer2' in name or 'layer3' in name or 'layer4' in name or 'fc' in name:
                parameter.requires_grad_(False)
        return_layers = {'layer1': 0}
        in_channels_list = in_channels_list[0:1]

    elif n_channel_backbone == 2 :
        for name, parameter in backbone.named_parameters():
            if 'layer1' not in name and 'layer2' not in name :
                parameter.requires_grad_(False)
        return_layers = {'layer1': 0, 'layer2': 1}
        in_channels_list = in_channels_list[0:2]

    elif n_channel_backbone == 3 :
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name :
                parameter.requires_grad_(False)
        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2}
        in_channels_list = in_channels_list[0:3]
    else :

        if first_layer_out == 'layer1' :

            for name, parameter in backbone.named_parameters():
                if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                   parameter.requires_grad_(False)

            # for name, parameter in backbone.named_parameters():
            #     if 'fc' in name :
            #        parameter.requires_grad_(False)

            return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

        elif first_layer_out == 'conv1' :

            for name, parameter in backbone.named_parameters():
                if 'fc' in name :
                   parameter.requires_grad_(False)

            return_layers = {'conv1': 0, 'layer1': 1, 'layer2': 2, 'layer3': 3}

            in_channels_list = [
                in_channels_stage2 // 4,
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
            ]

    out_channels = 256

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


def detnet_fpn_backbone(backbone_name, pretrained):

    backbone = detnet.__dict__[backbone_name](
        pretrained=pretrained)

    in_channels_stage2 = backbone.inplanes // 4

    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 4,
        in_channels_stage2 * 4,
    ]

    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name and 'layer5' not in name :
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'layer5': 4}

    out_channels = 256

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
