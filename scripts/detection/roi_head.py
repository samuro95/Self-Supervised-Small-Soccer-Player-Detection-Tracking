#Copyright (c) Soumith Chintala 2016, All rights reserved.

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



def fastrcnn_loss(class_logits, box_regression, labels, weights, regression_targets, weight_loss = False):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor])
    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """


    labels = torch.cat(labels, dim=0)
    weights = torch.cat(weights, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    if weight_loss :
        classification_loss_per_sample = F.cross_entropy(class_logits, labels, reduction='none')
        classification_loss = 0
        for i in range(len(classification_loss_per_sample)):
            classification_loss += classification_loss_per_sample[i]*weights[i]
        classification_loss/=weights.sum()
    else :
        classification_loss = F.cross_entropy(class_logits, labels)


    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    weight_pos = weights[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    # if weight_loss :
    #     box_loss_per_sample = F.smooth_l1_loss(
    #         box_regression[sampled_pos_inds_subset, labels_pos],
    #         regression_targets[sampled_pos_inds_subset],
    #         reduction="none").sum(dim = 1)
    #
    #     box_loss = 0
    #     for i in range(len(box_loss_per_sample)):
    #         box_loss += box_loss_per_sample[i]*weight_pos[i]
    # else :
    box_loss = F.smooth_l1_loss(
    box_regression[sampled_pos_inds_subset, labels_pos],
    regression_targets[sampled_pos_inds_subset],
    reduction="sum")

    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


# workaround for issue pytorch 27512
def tensor_floordiv(tensor, int_div):
    # type: (Tensor, int)
    result = tensor / int_div
    # TODO: https://github.com/pytorch/pytorch/issues/26731
    floating_point_types = (torch.float, torch.double, torch.half)
    if result.dtype in floating_point_types:
        result = result.trunc()
    return result


def _onnx_expand_boxes(boxes, scale):
    # type: (Tensor, float)
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half = w_half.to(dtype=torch.float32) * scale
    h_half = h_half.to(dtype=torch.float32) * scale

    boxes_exp0 = x_c - w_half
    boxes_exp1 = y_c - h_half
    boxes_exp2 = x_c + w_half
    boxes_exp3 = y_c + h_half
    boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
    return boxes_exp


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


class RoIHeads(torch.nn.Module):

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
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 # new
                 use_soft_nms = False,
                 weight_loss=False,
                 use_context=False,
                 use_track_branch = False,
                 track_embedding = None
                 ):

        super(RoIHeads, self).__init__()

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

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

        self.weight_loss = weight_loss
        self.use_soft_nms = use_soft_nms
        self.use_context = use_context

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

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
            labels_in_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1 # -1 is ignored by sampler

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
        if self.has_mask():
            assert self.DELTEME_all(["masks" in t for t in targets])

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

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]

        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            #print(boxes)
            if not self.use_soft_nms :
                keep = batched_nms(boxes, scores, labels, self.nms_thresh, self.use_soft_nms)
            else :
                boxes, scores, labels, keep = batched_nms(boxes, scores, labels, self.nms_thresh, self.use_soft_nms)
            #print(boxes)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

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

        if self.training :
            proposals, matched_idxs, labels, weights, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

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

            result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
            losses = {}

            if self.training:
                assert labels is not None and regression_targets is not None
                loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, weights, regression_targets, weight_loss = self.weight_loss)
                losses = {
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg
                }

            else:
                boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
                #print(boxes[0].shape)
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i],
                        }
                    )

            return result, losses
