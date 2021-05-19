
import sys
sys.path.append('../other_utils/metrics/')
from pascalvoc import compute_metrics
import cv2
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.utils
from faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from RPN import RPN
from faster_rcnn import FasterRCNN, fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn, fasterrcnn_detnet59_fpn, fasterrcnn_resnet8_fpn, fasterrcnn_resnet34_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from soccer_dataset import SoccerDataset
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import shutil
import numpy as np
from PIL import Image
import time
import pathlib


def enlargeBbox(bbox, scale, image_shape):
    min_x, min_y, max_x, max_y = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
    margin_x = int(0.5 * scale[0] * (max_x - min_x))
    margin_y = int(0.5 * scale[1] * (max_y - min_y))
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image_shape[2], max_x)
    max_y = min(image_shape[1], max_y)
    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def get_model_detection(model_name, weight_loss, backbone_name, pretrained,
                        pretrained_backbone, original, detection_score_thres, use_soft_nms,
                        anchor_sizes=[32, 64, 128, 256, 512], n_channel_backbone=5, use_context=False,
                        nms_thres=0.4, use_track_branch=False):

    num_classes = 2

    if model_name == 'frcnn_fpn':
        if backbone_name == 'resnet8':
            model = fasterrcnn_resnet8_fpn(weight_loss=weight_loss,
                                           pretrained_backbone=pretrained_backbone, num_classes=2,
                                           detection_score_thres=detection_score_thres, anchor_sizes=anchor_sizes,
                                           n_channel_backbone=n_channel_backbone, use_soft_nms=use_soft_nms,
                                           use_context=use_context)
        if backbone_name == 'resnet18':
            model = fasterrcnn_resnet18_fpn(weight_loss=weight_loss,
                                            pretrained_backbone=pretrained_backbone, num_classes=2,
                                            detection_score_thres=detection_score_thres, anchor_sizes=anchor_sizes,
                                            n_channel_backbone=n_channel_backbone, use_soft_nms=use_soft_nms,
                                            nms_thres=nms_thres,
                                            use_context=use_context,
                                            use_track_branch=use_track_branch)
        if backbone_name == 'resnet34':
            model = fasterrcnn_resnet34_fpn(weight_loss=weight_loss,
                                            pretrained_backbone=pretrained_backbone, num_classes=2,
                                            detection_score_thres=detection_score_thres, anchor_sizes=anchor_sizes,
                                            n_channel_backbone=n_channel_backbone, use_soft_nms=use_soft_nms,
                                            use_context=use_context)
        if backbone_name == 'resnet50':
            model = fasterrcnn_resnet50_fpn(weight_loss=weight_loss,
                                            pretrained=pretrained, pretrained_backbone=pretrained_backbone,
                                            num_classes=2, anchor_sizes=anchor_sizes,
                                            detection_score_thres=detection_score_thres,
                                            n_channel_backbone=n_channel_backbone, use_soft_nms=use_soft_nms,
                                            nms_thres=nms_thres,
                                            use_context=use_context, use_track_branch=use_track_branch)
        if backbone_name == 'detnet59':
            model = fasterrcnn_detnet59_fpn(weight_loss=weight_loss,
                                            pretrained=pretrained, pretrained_backbone=pretrained_backbone,
                                            num_classes=2, anchor_sizes=anchor_sizes,
                                            detection_score_thres=detection_score_thres,
                                            use_soft_nms=use_soft_nms, nms_thres=nms_thres, use_context=use_context)
        if not original:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == 'rpn':
        if backbone_name == 'resnet18':
            backbone_model = resnet_fpn_backbone('resnet18', pretrained_backbone)
            model = RPN(backbone_model, weight_loss=weight_loss)
        if backbone_name == 'resnet50':
            backbone_model = resnet_fpn_backbone('resnet50', pretrained_backbone)
            model = RPN(backbone_model, weight_loss=weight_loss)
            if pretrained:
                load_model = fasterrcnn_resnet50_fpn(pretrained=True)
                model.backbone = load_model.backbone
                model.rpn = load_model.rpn

    else:
        'model not available'

    return model


def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


def eval(model, test_loader, n_test, test_bs, device, n_update, original_model,
         writer=None, first_iter=0, get_image=False):

    rescale_bbox = [0., 0.]

    parts = list(pathlib.Path().absolute().parts)[:-2]
    parts.append('data')
    data_path = pathlib.Path(*parts)
    intermediate_path = os.path.join(data_path, 'intermediate', 'fasterRCNN_test')
    if not os.path.exists(intermediate_path):
        os.mkdir(intermediate_path)

    gt_txt_path = os.path.join(intermediate_path, 'GT')
    det_txt_path = os.path.join(intermediate_path, 'det')
    if os.path.exists(det_txt_path):
        shutil.rmtree(det_txt_path)
    os.mkdir(det_txt_path)
    if os.path.exists(gt_txt_path):
        shutil.rmtree(gt_txt_path)
    os.mkdir(gt_txt_path)

    test_loader_iterator = iter(test_loader)
    n_iter = min(n_test // test_bs, len(test_loader))
    model.eval()

    annotation_list = []
    detection_list = []
    scores_list = []

    time_list = []

    with torch.no_grad():

        for i in range(n_iter):

            for j in range(first_iter + 1):
                images, targets = next(test_loader_iterator)

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            t1 = time.time()
            detections = model(images)
            t2 = time.time()
            time_list.append(t2 - t1)

            if original_model:
                for j in range(len(images)):
                    annotation = targets[j]["boxes"].to(torch.device("cpu"))
                    detection = []
                    score = []
                    detection_all = detections[0][j]['boxes'].to(torch.device("cpu"))
                    label_all = detections[0][j]['labels'].to(torch.device("cpu"))
                    score_all = detections[0][j]['scores'].to(torch.device("cpu"))
                    for d, det in enumerate(detection_all):
                        if label_all[d] == 1:
                            detection.append(detection_all[d])
                            score.append(score_all[d])
                    annotation_list.append(annotation)
                    detection_list.append(detection)
                    scores_list.append(score)

            else:
                for j in range(len(images)):
                    annotation = targets[j]["boxes"].to(torch.device("cpu"))
                    detection = detections[0][j]['boxes'].to(torch.device("cpu"))
                    score = detections[0][j]['scores'].to(torch.device("cpu"))
                    annotation_list.append(annotation)
                    detection_list.append(detection)
                    scores_list.append(score)

            image_shape = images[0].shape

    for i in range(len(annotation_list)):
        with open(os.path.join(gt_txt_path, str(i) + '.txt'), 'w') as GT_file:
            with open(os.path.join(det_txt_path, str(i) + '.txt'), 'w') as det_file:
                annotation_boxes = annotation_list[i]
                det_boxes = detection_list[i]
                det_scores = scores_list[i]
                for bbox in annotation_boxes:
                    bbox = np.array(enlargeBbox(bbox, rescale_bbox, image_shape))
                    GT_file.write('person ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' +
                                  str(bbox[3]) + '\n')
                for j, bbox in enumerate(det_boxes):
                    det_score = det_scores[j].item()
                    det_file.write('person ' + str(det_score) + ' ' + str(bbox[0].item()) + ' ' + str(bbox[1].item()) +
                                   ' ' + str(bbox[2].item()) + ' ' + str(bbox[3].item()) + '\n')

    mAP = compute_metrics(gt_txt_path, det_txt_path, iouThreshold=0.5, showPlot=False)
    currentPath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(currentPath)

    if writer is not None or get_image:
        show_images = [cv2.cvtColor(np.array(image.cpu()).transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for image in images]
        out_tensor = []
        out_array = []
        for j in range(len(images)):
            detection = detections[0][j]['boxes'].to(torch.device("cpu"))
            label = detections[0][j]['labels'].to(torch.device("cpu"))
            annotation_boxes = targets[j]["boxes"].to(torch.device("cpu"))
            for d, bbox in enumerate(annotation_boxes):
                bbox = np.array(enlargeBbox(bbox, rescale_bbox, image_shape))
                cv2.rectangle(show_images[j],
                              (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                              (0, 0, 255), 2)
            for d, bbox in enumerate(detection):
                if label[d] == 1:
                    cv2.rectangle(show_images[j],
                                  (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                                  (255, 255, 255), 2)
            out_array.append(show_images[j])
            if writer is not None:
                shape = show_images[j].shape
                new_shape = (int(shape[1] / 4.), int(shape[0] / 4.))
                show_image = cv2.resize(show_images[j], dsize=new_shape, interpolation=cv2.INTER_CUBIC)
                show_image = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
                show_image = Image.fromarray((show_image * 255).astype(np.uint8))
                out_tensor.append(T.ToTensor()(show_image))
        if writer is not None:
            img_grid = torchvision.utils.make_grid(out_tensor)
            writer.add_image('val/imgs', img_grid, n_update)

    if get_image:
        return mAP, out_array
    else:
        return mAP


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--test_dataset_name', type=str, default='TV_soccer')
    parser.add_argument('--test_bs', type=int, default=4)
    parser.add_argument('--n_test', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--writer_iter', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='frcnn_fpn')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--scale_transform_test', type=float, default=1.)
    parser.add_argument('--rescale_method', type=str, default='pad')
    parser.add_argument('--detection_score_thres', type=float, default=0.05)
    parser.add_argument('--eval_original', dest='eval_original', action='store_true')
    parser.set_defaults(eval_original=False)
    parser.add_argument('--n_channel_backbone', type=int, default=5)
    parser.add_argument('--min_anchor', type=int, default=16)
    parser.add_argument('--use_hard_nms', dest='use_soft_nms', action='store_false')
    parser.set_defaults(use_soft_nms=True)
    parser.add_argument('--use_context', dest='use_context', action='store_true')
    parser.set_defaults(use_context=False)
    parser.add_argument('--use_field_detection', dest='use_field_detection', action='store_true')
    parser.set_defaults(use_field_detection=False)
    parser.add_argument('--use_track_branch', dest='use_track_branch', action='store_true')
    parser.set_defaults(use_track_branch=False)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2

    parts = list(pathlib.Path().absolute().parts)[:-2]
    parts.append('data')
    data_path = pathlib.Path(*parts)

    path = os.path.join(data_path, args.test_dataset_name)
    test_image_files = os.path.join(path, 'test_frame_list.txt')
    test_annotation_files = os.path.join(path, 'test_annotation_list.txt')

    dataset_test = SoccerDataset(
        test_image_files=test_image_files,
        test_annotation_files=test_annotation_files,
        transform=get_transform(),
        train=False,
        test_dataset_name=args.test_dataset_name,
        scale_transform=args.scale_transform_test,
        use_field_detection=args.use_field_detection)

    test_loader = DataLoader(
        dataset_test, batch_size=args.test_bs, shuffle=True, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    writer = SummaryWriter("runs/" + args.name)

    if args.min_anchor == 8:
        anchor_sizes = [8, 16, 32, 64, 128]
    elif args.min_anchor == 16:
        anchor_sizes = [16, 32, 64, 128, 256]
    else:
        anchor_sizes = [32, 64, 128, 256, 512]

    model = get_model_detection(args.model_name, False, args.backbone,
                                False, False, False, args.detection_score_thres,
                                args.use_soft_nms, anchor_sizes, args.n_channel_backbone,
                                args.use_context, use_track_branch=args.use_track_branch)
    model.to(device)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    if args.eval_original:
        model = get_model_detection(args.model_name, False, args.backbone,
                                    True, True, True, args.detection_score_thres, args.use_soft_nms)
        model.to(device)

    t1 = time.time()
    current_mAP = eval(model, test_loader, args.n_test, args.test_bs,
                       device, 0, args.eval_original)
    t2 = time.time()

    print('evalutation time (sec) : ', str(int(t2 - t1)))


# def extract_stats():
#     test_image_files = '../../data/TV_soccer/test_frame_list.txt'
#     test_annotation_files = '../../data/data/TV_soccer/test_annotation_list.txt'
#     rescale_bbox = [0., 0.]
#     dataset_test1 = SoccerDataset(test_image_files=test_image_files, test_annotation_files=test_annotation_files,
#                                   transform=get_transform(), train=False, data_name='TV_soccer', scale_transform=1.,
#                                   rescale_method='pad', use_field_detection=False)
#     test_loader1 = DataLoader(
#         dataset_test1, batch_size=1, shuffle=False, num_workers=0,
#         collate_fn=utils.collate_fn)
#
#     test_image_files = '../../data/issia/test_frame_list.txt'
#     test_annotation_files = '../../data/issia/test_annotation_list.txt'
#     dataset_test2 = SoccerDataset(test_image_files=test_image_files, test_annotation_files=test_annotation_files,
#                                   transform=get_transform(), train=False, data_name='issia', scale_transform=1.,
#                                   rescale_method='pad', use_field_detection=False)
#     test_loader2 = DataLoader(
#         dataset_test2, batch_size=1, shuffle=False, num_workers=0,
#         collate_fn=utils.collate_fn)
#
#     test_image_files = '../../data/SPD/test_frame_list.txt'
#     test_annotation_files = '../../data/SPD/test_annotation_list.txt'
#     dataset_test3 = SoccerDataset(test_image_files=test_image_files, test_annotation_files=test_annotation_files,
#                                   transform=get_transform(), train=False, data_name='SPD', scale_transform=1.,
#                                   rescale_method='pad', use_field_detection=False)
#     test_loader3 = DataLoader(
#         dataset_test3, batch_size=1, shuffle=False, num_workers=0,
#         collate_fn=utils.collate_fn)
#
#     dataset_train = SoccerDataset(transform=get_transform(),
#                                   train=True, weight_loss=False,
#                                   weight_seg=1.,
#                                   weight_track=0.,
#                                   scale_transform=1.,
#                                   rescale_method='pad',
#                                   image_shape=(720, 1280),
#                                   use_non_player=False,
#                                   only_det=False,
#                                   round_2=True,
#                                   data_name='SoccerNet',
#                                   train_annotation_files_r2='../../data/SoccerNet/train_annotation_list_r2_use_context.txt')
#
#     train_loader = DataLoader(
#         dataset_train, batch_size=1, shuffle=False, num_workers=0,
#         collate_fn=utils.collate_fn)
#
#     names = ['SoccerNet', 'TV soccer', 'ISSIA', 'SPD']
#
#     test_loaders = [train_loader, test_loader1, test_loader2, test_loader3]
#     out = []
#
#     for j in range(4):
#         print(j)
#
#         test_loader = test_loaders[j]
#         test_loader_iterator = iter(test_loader)
#
#         sum_h = 0
#         count = 0
#         sum_w = 0
#         sum_area = 0
#         area_tab = []
#
#         # c_det = 0
#         # c_seg = 0
#         # c_track = 0x
#         # c = 0
#         # for i in range(len(train_loader)) :
#         #     images, targets = next(train_loader_iterator)
#         #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets][0]
#         #     weights =  targets['weight']
#         #     for weight in weights :
#         #         c+=1
#         #         if weight == args.weight_seg :
#         #             c_seg +=1
#         #         elif weight == args.weight_track :
#         #             c_track+=1
#         #         else :
#         #             c_det+=1
#         #     print(c_det/c, c_seg/c, c_track/c)
#
#         if j == 0:
#             for i in range(len(test_loader))[:10000]:
#                 images, targets = next(test_loader_iterator)
#                 H, W = images[0].shape[1], images[0].shape[2]
#                 targets = [{k: v.to(device) for k, v in t.items()} for t in targets][0]
#                 boxes = targets['boxes'].cpu().numpy()
#                 for box in boxes:
#                     sum_w += (box[2] - box[0])
#                     sum_h += (box[3] - box[1])
#                     area_tab.append((box[2] - box[0]) * (box[3] - box[1]) * 1e6 / (H * W))
#                     # count += 1
#                 # print('w/W', sum_w / (W*count))
#                 # print('h/H', sum_h / (H*count))
#                 # print('area', sum_area / count)
#             out.append(area_tab)
#         else:
#             for i in range(len(test_loader)):
#                 images, targets = next(test_loader_iterator)
#                 H, W = images[0].shape[1], images[0].shape[2]
#                 targets = [{k: v.to(device) for k, v in t.items()} for t in targets][0]
#                 boxes = targets['boxes'].cpu().numpy()
#                 for box in boxes:
#                     sum_w += (box[2] - box[0])
#                     sum_h += (box[3] - box[1])
#                     area_tab.append((box[2] - box[0]) * (box[3] - box[1]) * 1e6 / (H * W))
#                     # count += 1
#                 # print('w/W', sum_w / (W*count))
#                 # print('h/H', sum_h / (H*count))
#                 # print('area', sum_area / count)
#             out.append(area_tab)
#
#     import matplotlib
#     from matplotlib import rcParams
#     rcParams.update({'figure.autolayout': True})
#     plt.rc('legend', fontsize=14)
#     plt.rc('axes', labelsize=14)
#     fig, ax1 = plt.subplots()
#     plt.hist(out, 50, density=True, label=names)
#     plt.xlabel('relat. bbox area')
#     plt.ylabel('proportion of players')
#     plt.legend()
#     ax1.set_xlim(0, 6000)
#     plt.grid(True)
#     plt.savefig('hist.png')