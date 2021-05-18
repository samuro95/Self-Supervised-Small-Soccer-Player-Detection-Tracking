# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from torchvision.models.detection.transform import resize_boxes
from collections import OrderedDict
import torch
from torch import nn

import matplotlib.pyplot as plt
import time


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, track_heads, transform, n_channel_backbone):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.track_heads = track_heads
        self.n_channel_backbone = n_channel_backbone

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if self.n_channel_backbone < 5 :
            in_channels = [(i, features[i]) for i in range(self.n_channel_backbone)]
            features = OrderedDict(in_channels)

        if self.n_channel_backbone > 5 :
            in_channels = [(i, features[i]) for i in range(5)]
            features = OrderedDict(in_channels)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        proposals, scores, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        for i, (pred, im_s, o_im_s) in enumerate(zip(proposals, images.image_sizes, original_image_sizes)):
            boxes = resize_boxes(pred, im_s, o_im_s)
            proposals[i] = boxes

        if self.training:
            return losses

        #return detections, proposals
        return detections, features
