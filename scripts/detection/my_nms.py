#Copyright (c) Soumith Chintala 2016, All rights reserved.

import torch
import torchvision
import time
import numpy as np

def nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torchvision.ops.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold, use_soft_nms):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # max_coordinate = boxes.max()
    # offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes

    if not use_soft_nms :
        #keep = py_cpu_nms(boxes_for_nms, scores, iou_threshold)
        keep = nms(boxes_for_nms, scores, iou_threshold)
        return keep
    else :
        boxes, scores = py_cpu_soft_nms(boxes_for_nms, scores, iou_thr=iou_threshold)
        labels = torch.ones_like(scores).to(torch.device('cuda'))
        keep = torch.tensor([i for i in range(len(labels))]).to(torch.device('cuda'))
        return boxes, scores, labels, keep


def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""

    dets = np.array(dets.cpu())
    scores = np.array(scores.cpu())
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def py_cpu_soft_nms(dets, scores, method='gaussian', iou_thr=0.3, sigma=0.4, score_thr=0.05):

    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    dets = np.array(dets.cpu())
    scores = np.array(scores.cpu())
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area

    dets = np.concatenate((dets, scores[:, None]), axis=1)
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []

    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    out = np.vstack(retained_box)
    boxes = torch.tensor(out[:,:-1]).to(torch.device('cuda'))
    scores = torch.tensor(out[:,-1]).to(torch.device('cuda'))

    return boxes, scores
