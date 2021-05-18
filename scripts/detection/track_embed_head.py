# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import division
import torch
import torchvision

import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops
from my_nms import batched_nms

from torchvision.ops import misc as misc_nn_ops

from torchvision.ops import roi_align

from  torchvision.models.detection import _utils as det_utils

from torch.jit.annotations import Optional, List, Dict, Tuple

import cv2

#from softnms_pytorch import soft_nms_pytorch



# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily for paste_mask_in_image
def expand_boxes(boxes, scale):
    # type: (Tensor, float)
    # if torchvision._is_tracing():
    #     return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


class TrackHeads(torch.nn.Module):

    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # new
                 weight_loss=False,
                 use_context=False,
                 track_embedding=None
                 ):

        super(TrackHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)

        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.weight_loss = weight_loss
        self.use_context = use_context
        self.track_embedding = track_embedding

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_weights):
        # type: (List[Tensor], List[Tensor], List[Tensor])
        matched_idxs = []
        labels = []
        weights = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_weights_in_image in zip(proposals, gt_boxes, gt_labels, gt_weights):
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            weights_in_image = gt_weights_in_image[clamped_matched_idxs_in_image]

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = torch.tensor(0)

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = torch.tensor(-1)  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            weights.append(weights_in_image)

        return matched_idxs, labels, weights

    def subsample(self, labels):
        # type: (List[Tensor])
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor])
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def DELTEME_all(self, the_list):
        # type: (List[bool])
        for i in the_list:
            if not i:
                return False
        return True

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]])
        assert targets is not None
        assert self.DELTEME_all(["boxes" in t for t in targets])
        assert self.DELTEME_all(["labels" in t for t in targets])

    def select_training_samples(self, proposals, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_weights = [t["weight"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels, weights = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_weights)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            weights[img_id] = weights[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, weights, regression_targets


    def forward(self, features, proposals, image_shapes, targets=None, get_feature_only = False):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'

        # if self.training :
        #     proposals, matched_idxs, labels, weights, regression_targets = self.select_training_samples(proposals, targets)
        # else:
        #     labels = None
        #     regression_targets = None
        #     matched_idxs = None

        #print(proposals)
        proposals = [proposals[0]]
        #print(proposals)

        box_features = self.box_roi_pool(features, proposals, image_shapes)

        if self.use_context :
            context_proposals = [expand_boxes(proposal, 4.) for proposal in proposals]
            context_features = self.box_roi_pool(features, context_proposals, image_shapes)
            #context_features = torch.zeros_like(context_features).to(torch.device('cuda'))
            box_features = torch.cat((box_features, context_features), 1)

        box_features_vect = self.box_head(box_features)

        if get_feature_only :
            return(box_features_vect)
        else :
            class_logits, box_regression = self.box_predictor(box_features_vect)
            track_embed = self.track_embedding(box_features)
            return track_embed
