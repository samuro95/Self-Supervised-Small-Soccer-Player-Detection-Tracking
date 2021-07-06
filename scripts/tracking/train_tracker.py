import sys
sys.path.append('../other_utils/metrics/')
from pascalvoc import compute_metrics
import cv2
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils
sys.path.append('../detection')
from faster_rcnn import FastRCNNPredictor
from faster_rcnn import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn, fasterrcnn_detnet59_fpn, fasterrcnn_resnet101_fpn, fasterrcnn_resnet34_fpn
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import numpy as np
from PIL import Image
import json
from natsort import natsorted, ns
from collections import defaultdict
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import random

def get_model_detection(model_name, weight_loss, backbone, pretrained,
    pretrained_backbone, original, detection_score_thres, use_non_player,
    use_soft_nms, anchor_sizes=[32,64,128,256,512], n_channel_backbone = 5,
    use_context=False, nms_thres = 0.4, use_track_branch = False):

        if use_non_player :
            num_classes = 3
        else :
            num_classes = 2

        if model_name == 'frcnn_fpn' :
            if backbone == 'resnet18' :
                model = fasterrcnn_resnet18_fpn(weight_loss=weight_loss,
                pretrained_backbone = pretrained_backbone, num_classes = 2,
                detection_score_thres = detection_score_thres, anchor_sizes = anchor_sizes,
                n_channel_backbone = n_channel_backbone, box_nms_thresh = nms_thres, use_soft_nms = use_soft_nms,
                use_context = use_context, use_track_branch = use_track_branch)
            if backbone == 'resnet50' :
                model = fasterrcnn_resnet50_fpn(weight_loss=weight_loss,
                pretrained = pretrained, pretrained_backbone = pretrained_backbone,
                num_classes=2, anchor_sizes = anchor_sizes, detection_score_thres = detection_score_thres,
                n_channel_backbone = n_channel_backbone, box_nms_thresh = nms_thres, use_soft_nms = use_soft_nms,
                use_context = use_context, use_track_branch = use_track_branch)
            if not original :
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        else :
            'model not available'
        return model

from main_tracking import track

def get_transform(train):
    transforms = []
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict

def rescale_img(img,image_shape,current_scale_transform):
    w = image_shape[2]
    h = image_shape[1]
    desired_h = h*current_scale_transform
    desired_w = w*current_scale_transform
    img = torchvision.transforms.Resize([int(desired_h), int(desired_w)])(img)
    w_pad = (w - w*current_scale_transform)/2.
    h_pad = (h - h*current_scale_transform)/2.
    img = torchvision.transforms.Pad((int(w_pad),int(h_pad)))(img)
    return(img)

def rescale_bbox(bbox,image_shape,current_scale_transform):
    w = image_shape[2]
    h = image_shape[1]
    bbox = np.array(bbox)*current_scale_transform
    target_w = w*current_scale_transform
    target_h = h*current_scale_transform
    w_pad = (w - target_w)/2.
    h_pad = (h - target_h)/2.
    new_w = target_w + 2*w_pad
    new_h = target_h + 2*h_pad
    bbox_center = bbox + np.array([w_pad,h_pad,w_pad,h_pad])
    return(bbox_center)

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: torch.Tensor with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0)).byte()
    print(indices_equal)
    if labels.is_cuda:
        indices_equal = indices_equal.cuda()
    indices_not_equal = ~indices_equal
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    print(indices_not_equal)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    #if labels.is_cuda:
    #    label_equal = label_equal.cuda()
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = i_equal_j & (~i_equal_k)

    # Combine the two masks
    mask = distinct_indices & valid_labels

    return mask

def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: torch.Tensor with shape [batch_size]
    Returns:
        mask: Varieble with torch.ByteTensor with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0)).bool()
    if labels.is_cuda:
        indices_equal = indices_equal.cuda()
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    # Combine the two masks
    mask = indices_not_equal & labels_equal

    return mask

def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: torch.Tensor with shape [batch_size]
    Returns:
        mask: Variable with torch.ByteTensor with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    mask = ~labels_equal
    return mask


def extract_track_list(annotation_path,n_train,init_train) :
    track_list = defaultdict(list)
    with open(annotation_path,'r') as ann_file :
        lines = ann_file.readlines()
        for line in lines :
            line = line.split(',')
            img_id, track_id, xmin,ymin,xmax,ymax = line[:6]
            if init_train <= int(img_id) < init_train+n_train :
                track_list[img_id].append({'track_id' : track_id, 'bbox' : [float(xmin),float(ymin),float(xmax),float(ymax)]})
    return(track_list)


def extract_eval_track_list(annotation_path,n_train,init_train) :
    track_list = defaultdict(list)
    with open(annotation_path,'r') as ann_file :
        lines = ann_file.readlines()
        for line in lines :
            line = line.split(',')
            img_id, track_id, xmin,ymin,xmax,ymax = line[:6]
            if init_train <= int(img_id) < init_train+n_train :
                track_list[img_id].append({'track_id' : track_id, 'bbox' : [float(xmin),float(ymin),float(xmax),float(ymax)]})
    return(track_list)


# def eval_triplet_loss(eval_annotation_path,n_test,n_max_track_eval,rescale_img_factor,image_shape,eval_annotation_path = '/home/pvitoria/Documents/ARPE/soccer_tracking/data/issia/annotations/0/'):
#
#     annotation_paths = natsorted(os.listdir(annotation_folder), alg=ns.PATH | ns.IGNORECASE)
#     track_list = extract_val_track_list(eval_annotation_path, n_test)
#     track_dict = defaultdict(list)
#     for k,v in track_list.items():
#         for el in v :
#             track_id = el['track_id']
#             track_dict[track_id].append({'img_id' : int(k), 'bbox' : el['bbox'], 'embed' : el['embed']})
#
#     for path in natsorted(os.listdir(eval_annotation_path), alg=ns.PATH | ns.IGNORECASE)[:n_test] :
#
#         proposal_bboxs = []
#         for k in range(len(track_list[str(i)])) :
#             bbox = track_list[str(i)][k]['bbox']
#             bbox = rescale_bbox(bbox,image_shape,current_scale_transform)
#             track_list[str(i)][k]['bbox'] = bbox
#             proposal_bboxs.append(bbox)
#
#         prop = {}
#         prop["boxes"] = torch.as_tensor(proposal_bboxs, dtype=torch.float32)
#         proposals = [prop]
#         proposals = [{k: v.to(device) for k, v in t.items()} for t in proposals]
#
#         images, targets = model.transform(images_in, targets)
#         images, proposals_in = model.transform(images_in, proposals)
#         proposals_in = [proposals[0]["boxes"]]
#
#         features = model.backbone(images.tensors)
#
#         n_prop = len(proposals_in[0])
#
#         detector_losses_tab = []
#         proposal_losses_tab = []
#
#         if args.use_track_branch :
#             proposals_rpn, scores, proposal_losses = model.rpn(images, features, targets)
#             _, detector_losses = model.roi_heads(features, proposals_rpn, images.image_sizes, targets)
#             detector_losses_tab.append(sum(loss for loss in detector_losses.values()))
#             proposal_losses_tab.append(sum(loss for loss in proposal_losses.values()))
#             track_embed = model.track_heads(features, proposals_in, images.image_sizes, targets)
#         else :
#             proposals_rpn, scores, proposal_losses = model.rpn(images, features, targets)
#             _, detector_losses = model.roi_heads(features, proposals_rpn, images.image_sizes, targets)
#             track_embed = model.roi_heads(features, proposals_in, images.image_sizes, targets, get_feature_only=True)
#             detector_losses_tab.append(sum(loss for loss in detector_losses.values()))
#             proposal_losses_tab.append(sum(loss for loss in proposal_losses.values()))
#
#         track_embed = track_embed[:n_prop]
#
#         if args.visualize :
#             imshow = np.array(img).transpose(1, 2, 0)
#             imshow = cv2.cvtColor(imshow, cv2.COLOR_RGB2BGR)*255
#             vis_tab.append(imshow)
#             #cv2.imshow('c',imshow)
#             #cv2.waitKey(100)
#
#         for k in range(n_prop):
#             if args.visualize :
#                 det = proposals_in[0][k]
#                 cv2.rectangle(imshow, (int(det[0]), int(det[1]),int(det[2]-det[0]), int(det[3]-det[1])), (255, 255, 255), 2)
#             track_list[str(i)][k]['embed'] = track_embed[k]
#
#     for t,track_id in enumerate(track_dict.keys()):
#
#         if t < n_max_track_eval :
#
#             res = []
#             labels = []
#
#             pos = np.array(track_dict[track_id])
#
#             if len(pos)>K :
#
#                 pos_samples = np.random.choice(len(pos), K, replace=False)
#                 res.append(pos[pos_samples])
#                 labels = [int(track_id)] * K
#
#                 img_ids_used = []
#                 img_ids_pos = [pos[id]['img_id'] for id in pos_samples]
#                 img_ids_used.append(img_ids_pos)
#
#                 neg_indices = []
#                 key_list = list(track_dict.keys())
#                 idd = 0
#                 while len(neg_indices) < P-1 and idd < len(key_list):
#                     idx = key_list[idd]
#                     if idx != track_id and len(track_dict[idx]) > K :
#                         neg = track_dict[idx]
#                         img_ids_neg = [neg[id]['img_id'] for id in range(len(neg))]
#                         b = True
#                         for id_list in img_ids_used :
#                             if len(set(img_ids_neg).intersection(set(id_list))) == 0 :
#                                 b = False
#                         if b :
#                             neg_indices.append(idx)
#                             img_ids_used.append(img_ids_neg)
#                     idd+=1
#
#                 if len(neg_indices) < 2 :
#                     continue
#
#                 for idx in neg_indices:
#                     neg = np.array(track_dict[idx])
#                     res.append(neg[np.random.choice(len(neg), K, replace=False)])
#                     labels += [int(idx)] * K
#
#             labels = torch.tensor(labels)
#
#             embeddings = []
#             for player in res :
#                 for el in player :
#                     embed = el['embed']
#                     embeddings.append(embed)
#             if len(embeddings) > 0 :
#                 embeddings = torch.stack(embeddings,0)
#
#             if len(embeddings) > 0 :
#
#                 mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float().to(device)
#                 mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float().to(device)
#
#                 n = embeddings.size(0)
#                 m = embeddings.size(0)
#                 d = embeddings.size(1)
#
#                 x = embeddings.data.unsqueeze(1).expand(n, m, d)
#                 y = embeddings.data.unsqueeze(0).expand(n, m, d)
#
#                 dist = torch.pow(x - y, 2).sum(2)
#
#                 pos_dist = dist * mask_anchor_positive
#                 max_val = torch.max(dist)
#                 neg_dist = dist + max_val * (1.0 - mask_anchor_negative)
#
#                 # for each anchor compute hardest pair
#                 triplets = []
#                 for i in range(dist.size(0)):
#                     pos = torch.max(pos_dist[i],0)[1].item()
#                     neg = torch.min(neg_dist[i],0)[1].item()
#                     triplets.append((i, pos, neg))
#
#                 e0 = []
#                 e1 = []
#                 e2 = []
#                 for p in triplets:
#                     e0.append(embeddings[p[0]])
#                     e1.append(embeddings[p[1]])
#                     e2.append(embeddings[p[2]])
#                 e0 = torch.stack(e0,0)
#                 e1 = torch.stack(e1,0)
#                 e2 = torch.stack(e2,0)
#
#                 t_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=2, reduction = 'mean')
#                 loss = t_loss
#
#
#                 loss_tab.append(loss)
#
#             loss_track = sum(l for l in loss_tab)
#
#             sum_loss_track += loss_track
#             loss_n +=1
#             avg_loss_track = sum_loss_track / loss_n
#             avg_track_loss_float = float(avg_loss_track.to(torch.device("cpu")))
#             try :
#                 track_loss_float = float(loss_track.to(torch.device("cpu")))
#             except :
#                 track_loss_float = 0.


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default = 'margin0.5', help = 'name of the experiment')
    parser.add_argument('--root_annotation', type=str, default = '../../data/tracking/SoccerNet')
    parser.add_argument('--root_bbox_annotation', type=str, default = '../../data/SoccerNet/annotations_r2/')
    parser.add_argument('--train', type=str, default = 'True')
    parser.add_argument('--n_train', type=int, default = 20)
    parser.add_argument('--n_test', type=int, default = 8)
    parser.add_argument('--epoch', type=int, default = 20)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--only_det', type=str, default = 'False')
    parser.add_argument('--model_name', type=str, default = 'frcnn_fpn')
    parser.add_argument('--backbone', type=str, default = 'resnet18')
    parser.add_argument('--checkpoint', type=str, default = '../../checkpoints_runs/RCNN/resnet18_student.pth')
    parser.add_argument('--start_epoch', type=int, default = 0)
    parser.add_argument('--scale_transform_train', type=float, default = 0.1)
    parser.add_argument('--scale_transform_test', type=float, default = 0.5)
    parser.add_argument('--rescale_method', type=str, default = 'pad')
    parser.add_argument('--detection_score_thres', type=float, default = 0.05)
    parser.add_argument('--eval_original', type=str, default = 'False')
    parser.add_argument('--use_non_player', type=str, default = 'False')
    parser.add_argument('--n_channel_backbone', type=int, default = 5)
    parser.add_argument('--min_anchor', type=int, default = 16)
    parser.add_argument('--use_soft_nms', type=str, default = 'True')
    parser.add_argument('--use_context', type=str, default = 'True')
    parser.add_argument('--weight_loss', type=str, default = 'False')
    parser.add_argument('--val_iter', type=int, default = 50)
    parser.add_argument('--use_track_branch', type=str, default = 'False')
    parser.add_argument('--train_backbone', type=str, default = 'True')
    parser.add_argument('--visualize', type=str, default = 'False')

    args = parser.parse_args()
    print(args)

    args.use_non_player  = (args.use_non_player == 'True')
    args.eval_original  = (args.eval_original == 'True')
    args.train  = (args.train == 'True')
    args.use_soft_nms  = (args.use_soft_nms == 'True')
    args.use_context  = (args.use_context == 'True')
    args.only_det  = (args.only_det == 'True')
    args.weight_loss  = (args.weight_loss == 'True')
    args.use_track_branch  = (args.use_track_branch == 'True')
    args.train_backbone  = (args.train_backbone == 'True')
    args.visualize  = (args.visualize == 'True')

    save_dir = 'checkpoint/' + args.name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    runs_dir = "runs/"+args.name

    writer = SummaryWriter(runs_dir)

    with open(runs_dir + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.min_anchor == 8 :
        anchor_sizes = [8,16,32,64,128]
    elif args.min_anchor == 16 :
        anchor_sizes = [16,32,64,128,256]
    else :
        anchor_sizes = [32,64,128,256,512]

    model = get_model_detection(args.model_name, args.weight_loss, args.backbone,
    False, False, False, args.detection_score_thres,
    args.use_non_player, args.use_soft_nms, anchor_sizes, args.n_channel_backbone,
    args.use_context, use_track_branch = args.use_track_branch)
    model.train()

    model.to(device)

    if args.checkpoint is not None :
       model.load_state_dict(torch.load(args.checkpoint),strict=False)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    rpn_params = [p for p in model.rpn.parameters() if p.requires_grad]
    roi_params = [p for p in model.roi_heads.parameters() if p.requires_grad]
    if args.use_track_branch :
        track_head_params = [p for p in model.track_heads.parameters() if p.requires_grad]
        if args.train_backbone :
            params = track_head_params + backbone_params + roi_params + rpn_params
        else :
            params = track_head_params
    else :
        if args.train_backbone :
            params = roi_params + rpn_params + backbone_params
        else :
            params = roi_params

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    #optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.000)

    n_update = 0
    loss_n = 0
    sum_loss_track = 0

    transforms = []
    transforms.append(T.ToTensor())
    transform = T.Compose(transforms)

    image_shape = [3,720,1280]
    K = 5
    P = 10
    margin = 0.5
    n_max_track_batch = 1

    for epoch in range(args.epoch):

        torch.save(model.state_dict(), save_dir + '/' + str(epoch) + '.pth')
        model.train()

        for s in natsorted(os.listdir(args.root_annotation), alg=ns.PATH | ns.IGNORECASE) :

            current_scale_transform = random.uniform(args.scale_transform_train,1.)
            init_train = int(random.uniform(0, 120 - args.n_train))
            model.zero_grad()

            print('seq ',s)

            annotation_path = os.path.join(args.root_annotation,s,'output_tracking','out.csv')
            track_list = extract_track_list(annotation_path,args.n_train,init_train)
            bbox_annotation_path =  os.path.join(args.root_bbox_annotation,s)

            vis_tab = []

            try :

                for path in natsorted(os.listdir(bbox_annotation_path), alg=ns.PATH | ns.IGNORECASE)[init_train:init_train+args.n_train] :

                    i = int(path[:-4])

                    target = parse_voc_xml(ET.parse(os.path.join(bbox_annotation_path,path)).getroot())
                    anno = target['annotation']

                    img_path = anno['path']
                    if 'u176443' in img_path :
                        img_path = img_path.replace('u176443','pvitoria')
                    img = Image.open(img_path).convert('RGB')

                    img = rescale_img(img,image_shape,current_scale_transform)
                    img = transform(img)

                    h, w = anno['size']['height'], anno['size']['width']
                    boxes = []
                    classes = []
                    area = []
                    weight = []
                    iscrowd = []

                    try :
                        objects = anno['object']
                    except :
                        objects = []

                    for obj in objects :
                        bbox = obj['bndbox']
                        w = float(obj["difficult"])
                        id =  obj['name']
                        if not False in [el in bbox.keys() for el in ['xmin', 'ymin', 'xmax', 'ymax']] :
                            try :
                                bbox = [int(float(bbox[n])) for n in ['xmin', 'ymin', 'xmax', 'ymax']]
                            except :
                                bbox = [int(float(bbox[n][7:-1])) for n in ['xmin', 'ymin', 'xmax', 'ymax']]

                            if w > 0.8 and id == 'player' :
                                bbox = rescale_bbox(bbox,image_shape,current_scale_transform)
                                boxes.append(bbox)
                                weight.append(1.)
                                classes.append(1)
                                area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                                iscrowd.append(False)

                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                    classes = torch.as_tensor(classes)
                    area = torch.as_tensor(area)
                    iscrowd = torch.as_tensor(iscrowd)
                    weight = torch.as_tensor(weight)

                    output = {}
                    output["boxes"] = boxes
                    output["weight"] = weight
                    output["labels"] = classes
                    output["area"] = area
                    output["iscrowd"] = iscrowd

                    device = torch.device('cuda')

                    images_in = [img]
                    targets = [output]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    images_in = list(image.to(device) for image in images_in)

                    proposal_bboxs = []
                    for k in range(len(track_list[str(i)])) :
                        bbox = track_list[str(i)][k]['bbox']
                        bbox = rescale_bbox(bbox,image_shape,current_scale_transform)
                        track_list[str(i)][k]['bbox'] = bbox
                        proposal_bboxs.append(bbox)

                    prop = {}
                    prop["boxes"] = torch.as_tensor(proposal_bboxs, dtype=torch.float32)
                    proposals = [prop]
                    proposals = [{k: v.to(device) for k, v in t.items()} for t in proposals]

                    images, targets = model.transform(images_in, targets)
                    images, proposals_in = model.transform(images_in, proposals)
                    proposals_in = [proposals[0]["boxes"]]

                    features = model.backbone(images.tensors)

                    n_prop = len(proposals_in[0])

                    detector_losses_tab = []
                    proposal_losses_tab = []

                    if args.use_track_branch :
                        proposals_rpn, scores, proposal_losses = model.rpn(images, features, targets)
                        _, detector_losses = model.roi_heads(features, proposals_rpn, images.image_sizes, targets)
                        detector_losses_tab.append(sum(loss for loss in detector_losses.values()))
                        proposal_losses_tab.append(sum(loss for loss in proposal_losses.values()))
                        track_embed = model.track_heads(features, proposals_in, images.image_sizes, targets)
                    else :
                        proposals_rpn, scores, proposal_losses = model.rpn(images, features, targets)
                        _, detector_losses = model.roi_heads(features, proposals_rpn, images.image_sizes, targets)
                        track_embed = model.roi_heads(features, proposals_in, images.image_sizes, targets, get_feature_only=True)
                        detector_losses_tab.append(sum(loss for loss in detector_losses.values()))
                        proposal_losses_tab.append(sum(loss for loss in proposal_losses.values()))

                    track_embed = track_embed[:n_prop]

                    if args.visualize :
                        imshow = np.array(img).transpose(1, 2, 0)
                        imshow = cv2.cvtColor(imshow, cv2.COLOR_RGB2BGR)*255
                        vis_tab.append(imshow)
                        #cv2.imshow('c',imshow)
                        #cv2.waitKey(100)

                    for k in range(n_prop):
                        if args.visualize :
                            det = proposals_in[0][k]
                            cv2.rectangle(imshow, (int(det[0]), int(det[1]),int(det[2]-det[0]), int(det[3]-det[1])), (255, 255, 255), 2)
                        track_list[str(i)][k]['embed'] = track_embed[k]

                    if args.visualize :
                        cv2.imwrite('vis/'+str(i)+'.png',imshow)
                    #     cv2.imshow('k',imshow/255)
                    #     cv2.waitKey(100)

            except Exception as e :
                print(e)
                if e is not KeyboardInterrupt :
                    continue

            proposal_loss = sum(l for l in proposal_losses_tab)
            detector_loss = sum(l for l in detector_losses_tab)
            detection_loss = detector_loss + proposal_loss

            track_dict = defaultdict(list)
            for k,v in track_list.items():
                for el in v :
                    track_id = el['track_id']
                    track_dict[track_id].append({'img_id' : int(k), 'bbox' : el['bbox'], 'embed' : el['embed']})

            loss_tab = []

            for t,track_id in enumerate(track_dict.keys()):

                if t < n_max_track_batch :

                    res = []
                    labels = []

                    pos = np.array(track_dict[track_id])

                    if len(pos)>K :

                        pos_samples = np.random.choice(len(pos), K, replace=False)
                        res.append(pos[pos_samples])
                        labels = [int(track_id)] * K

                        img_ids_used = []
                        img_ids_pos = [pos[id]['img_id'] for id in pos_samples]
                        img_ids_used.append(img_ids_pos)

                        neg_indices = []
                        key_list = list(track_dict.keys())
                        idd = 0
                        while len(neg_indices) < P-1 and idd < len(key_list):
                            idx = key_list[idd]
                            if idx != track_id and len(track_dict[idx]) > K :
                                neg = track_dict[idx]
                                img_ids_neg = [neg[id]['img_id'] for id in range(len(neg))]
                                b = True
                                for id_list in img_ids_used :
                                    if len(set(img_ids_neg).intersection(set(id_list))) == 0 :
                                        b = False
                                if b :
                                    neg_indices.append(idx)
                                    img_ids_used.append(img_ids_neg)
                            idd+=1

                        if len(neg_indices) < 2 :
                            continue

                        for idx in neg_indices:
                            neg = np.array(track_dict[idx])
                            res.append(neg[np.random.choice(len(neg), K, replace=False)])
                            labels += [int(idx)] * K

                    labels = torch.tensor(labels)

                    embeddings = []
                    for player in res :
                        for el in player :
                            embed = el['embed']
                            embeddings.append(embed)
                    if len(embeddings) > 0 :
                        embeddings = torch.stack(embeddings,0)

                    if len(embeddings) > 0 :

                        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float().to(device)
                        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float().to(device)

                        n = embeddings.size(0)
                        m = embeddings.size(0)
                        d = embeddings.size(1)

                        x = embeddings.data.unsqueeze(1).expand(n, m, d)
                        y = embeddings.data.unsqueeze(0).expand(n, m, d)

                        dist = torch.pow(x - y, 2).sum(2)

                        pos_dist = dist * mask_anchor_positive
                        max_val = torch.max(dist)
                        neg_dist = dist + max_val * (1.0 - mask_anchor_negative)

                        # for each anchor compute hardest pair
                        triplets = []
                        for i in range(dist.size(0)):
                            pos = torch.max(pos_dist[i],0)[1].item()
                            neg = torch.min(neg_dist[i],0)[1].item()
                            triplets.append((i, pos, neg))

                        e0 = []
                        e1 = []
                        e2 = []
                        for p in triplets:
                            e0.append(embeddings[p[0]])
                            e1.append(embeddings[p[1]])
                            e2.append(embeddings[p[2]])
                        e0 = torch.stack(e0,0)
                        e1 = torch.stack(e1,0)
                        e2 = torch.stack(e2,0)

                        t_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=2, reduction = 'mean')
                        #pos_loss = F.mse_loss(e0,e1)
                        #neg_loss = F.mse_loss(e0,e2)
                        #print(np.sqrt(pos_loss.cpu().detach().numpy())-np.sqrt(neg_loss.cpu().detach().numpy()))
                        loss = t_loss

                        # optimizer.zero_grad()
                        # loss.backward(retain_graph=True)

                        loss_tab.append(loss)

                        # c = 0
                        # loss_tab.append(total_loss)
                        # fl = float(total_loss.to(torch.device("cpu")))
                        # #print(fl)
                        # if fl > 1. :
                        #     if args.visualize :
                        #         c+=1
                        #         for p,player in enumerate(res[:1]) :
                        #             for i,el in enumerate(player) :
                        #                 img_id = el['img_id']
                        #                 det = el['bbox']
                        #                 imshow = vis_tab[img_id]/255.
                        #                 cv2.rectangle(imshow, (int(det[0]), int(det[1]),int(det[2]-det[0]), int(det[3]-det[1])), (255, 0, 0), 2)
                        #                 #cv2.imwrite('vis/'+str(p)+'_'+str(i)+'.png',imshow)
                        #                 cv2.imshow(str(p),imshow)
                        #                 cv2.waitKey(1000)
                        #         cv2.destroyAllWindows()

            loss_track = sum(l for l in loss_tab)

            if args.train_backbone :
                L = loss_track + detection_loss
            else :
                L = loss_track

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            sum_loss_track += loss_track
            loss_n +=1
            avg_loss_track = sum_loss_track / loss_n
            proposal_loss_float = float(proposal_loss.to(torch.device("cpu")))
            detector_loss_float = float(detector_loss.to(torch.device("cpu")))
            avg_track_loss_float = float(avg_loss_track.to(torch.device("cpu")))
            try :
                track_loss_float = float(loss_track.to(torch.device("cpu")))
            except :
                track_loss_float = 0.

            print('total track loss', track_loss_float)
            print('total detector loss', detector_loss_float)
            print('total proposal loss', proposal_loss_float)
            optimizer.step()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            writer.add_scalar('train/avg_loss_track', avg_track_loss_float, n_update)
            writer.add_scalar('train/track_loss', track_loss_float, n_update)
            writer.add_scalar('train/detector_loss', detector_loss_float, n_update)
            writer.add_scalar('train/proposal_loss', proposal_loss_float, n_update)
            n_update+=1

            ### evaluate tracking
            if int(s)%args.val_iter == 0 :

                model.eval()
                with torch.no_grad() :
                    try :
                        mota, motp, idf1 = track(data_name = 'issia',
                            model_name = 'frcnn_fpn',
                            current_model_detection = model,
                            detection_score_thres = 0.8,
                            weight_loss = args.weight_loss,
                            use_context = args.use_context,
                            anchor_sizes = anchor_sizes,
                            use_track_branch_model = args.use_track_branch,
                            use_track_branch_embed = args.use_track_branch,
                            n_img_max = 50,
                            use_GT_position = False,
                            w_spacial = 0.97,
                            w_visual = 0.03,
                            use_IOU = True,
                            use_features = True,
                            use_visual_feat = True,
                            write_csv = False,
                            write_video = True,
                            visual_feat_model_name = 'faster-rcnn', #or 'vgg' or 'faster-rcnn'
                            imagenet_model = False,
                            visualize = False,
                            rescale_img_factor = 0.5,
                            use_filter_tracks = False,
                            thres_count_ids = 2,
                            use_soft_nms = True,
                            nms_thres = 0.8,
                            N_frame_lost_keep = 10,
                            N_past_to_keep = 1,
                            N_past_to_keep_reID = 5,
                            max_vis_feat = 4.,
                            max_vis_reID = 8.,
                            use_ReID_module = True,
                            visual_metric = 'l2')

                        writer.add_scalar('val/idf1', idf1, n_update)
                        writer.add_scalar('val/mota', mota, n_update)
                        writer.add_scalar('val/motp', motp, n_update)

                    except Exception as e :
                        print(e)
                    model.train()
