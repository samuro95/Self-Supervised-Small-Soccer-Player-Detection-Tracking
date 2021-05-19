import sys
sys.path.append('../other_utils/metrics/')
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
from faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from RPN import RPN
from faster_rcnn import FasterRCNN, fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn, fasterrcnn_detnet59_fpn, fasterrcnn_resnet101_fpn, fasterrcnn_resnet34_fpn, fasterrcnn_resnet8_fpn
from soccer_dataset import SoccerDataset
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
from eval_fasterRCNN import eval
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default = 'test', help = 'name of the experiment')
    parser.add_argument('--train', type=str, default = 'True')
    parser.add_argument('--train_bs', type=int, default = 4)
    parser.add_argument('--test_bs', type=int, default = 4)
    parser.add_argument('--n_test', type=int, default = 400)
    parser.add_argument('--epoch', type=int, default = 20)
    parser.add_argument('--lr', type=float, default = 0.001)
    parser.add_argument('--num_workers', type=int, default = 8)
    parser.add_argument('--writer_iter', type=int, default = 10)
    parser.add_argument('--eval_iter', type=int, default = 200)
    parser.add_argument('--save_iter', type=int, default = 200)
    parser.add_argument('--test_dataset_name', type=str, default = 'TV_soccer')
    parser.add_argument('--weight_loss', dest='weight_loss', action='store_true')
    parser.set_defaults(weight_loss=False)
    parser.add_argument('--weight_seg', type=float, default=1.)
    parser.add_argument('--weight_track', type=float, default=1.)
    parser.add_argument('--only_det', dest='only_det', action='store_true')
    parser.set_defaults(only_det=False)
    parser.add_argument('--model_name', type=str, default = 'frcnn_fpn')
    parser.add_argument('--backbone', type=str, default = 'resnet50')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.set_defaults(pretrained=False)
    parser.add_argument('--not_pretrained_backbone', dest='pretrpretrained_backboneained', action='store_false')
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument('--checkpoint', type=str, default = None)
    parser.add_argument('--start_epoch', type=int, default = 0)
    parser.add_argument('--scale_transform_train', type=float, default = 1.)
    parser.add_argument('--scale_transform_test', type=float, default = 1.)
    parser.add_argument('--detection_score_thres', type=float, default = 0.05)
    parser.add_argument('--n_channel_backbone', type=int, default = 5)
    parser.add_argument('--min_anchor', type=int, default = 16)
    parser.add_argument('--first_iter_test', type=int, default = 0)
    parser.add_argument('--use_hard_nms', dest='use_soft_nms', action='store_false')
    parser.set_defaults(use_soft_nms=True)
    parser.add_argument('--use_context', dest='use_context', action='store_true')
    parser.set_defaults(use_context=False)
    parser.add_argument('--scheduler_step_1', type=int, default = 8)
    parser.add_argument('--scheduler_step_2', type=int, default = 14)
    parser.add_argument('--scheduler_step_3', type=int, default = 17)
    parser.add_argument('--round_2', dest='round_2', action='store_true')
    parser.set_defaults(round_2=False)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2

    parts = list(pathlib.Path().absolute().parts)[:-2]
    path = pathlib.Path(*parts)
    checkpoints_runs_dir = os.path.join(path, 'checkpoints_runs', 'RCNN')
    checkpoints_dir = os.path.join(checkpoints_runs_dir, args.name)
    runs_dir = os.path.join(checkpoints_runs_dir, args.name)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.isdir(runs_dir):
        os.makedirs(runs_dir)

    if args.train :

        dataset_train = SoccerDataset(transform = get_transform(),
                    train = True, weight_loss = args.weight_loss,
                    weight_seg = args.weight_seg,
                    weight_track = args.weight_track,
                    scale_transform = args.scale_transform_train,
                    only_det = args.only_det,
                    round_2 = args.round_2,
                    visualize=False)

        train_loader = DataLoader(
            dataset_train, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers,
            collate_fn=utils.collate_fn)

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
        visualize=False)

    test_loader = DataLoader(
        dataset_test, batch_size=args.test_bs, shuffle=True, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

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
            args.pretrained, args.pretrained_backbone, False, args.detection_score_thres,
            args.use_soft_nms, anchor_sizes, args.n_channel_backbone,
            args.use_context)

    model.to(device)

    if args.checkpoint is not None :
        model.load_state_dict(torch.load(args.checkpoint))

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    rpn_params = [p for p in model.rpn.parameters() if p.requires_grad]
    roi_params = [p for p in model.roi_heads.parameters() if p.requires_grad]
    params = backbone_params + rpn_params + roi_params

    optimizer = torch.optim.SGD(params, lr=args.lr,momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[args.scheduler_step_1,args.scheduler_step_1,args.scheduler_step_1], gamma=0.1)


    n_update = 0
    loss_n = 0
    loss_sum = 0

    n_iter = len(train_loader)

    for epoch in range(args.start_epoch):
        for i in range(n_iter):
            n_update +=1
            optimizer.step()
        lr_scheduler.step()

    for epoch in range(args.start_epoch,args.epoch):

        torch.save(model.state_dict(), checkpoints_dir + '/' + str(epoch) + '.pth')

        train_loader_iterator = iter(train_loader)

        train_loader = tqdm(train_loader)

        current_mAP = eval(model, test_loader, args.n_test, args.test_bs, device,
                           n_update, False, writer = writer)

        model.train()

        for i in range(n_iter):

            lr = optimizer.param_groups[0]['lr']

            n_update +=1
            model.zero_grad() #clears old gradients from the last step

            images, targets = next(train_loader_iterator)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            loss_objectness = loss_dict['loss_objectness']
            loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']
            loss_classifier = loss_dict['loss_classifier']
            loss_box_reg = loss_dict['loss_box_reg']

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            import math
            if math.isnan(float(loss_value)) :
                continue

            loss_n += len(images)
            loss_sum += float(loss_value)
            avg_loss = loss_sum / loss_n

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (i+1) % args.writer_iter == 0 :
                writer.add_scalar('train/lr', lr, n_update)
                for loss in loss_dict.keys() :
                    writer.add_scalar('train/train_loss', float(losses.to(torch.device("cpu"))), n_update)
                    writer.add_scalar('train/' + loss, float(loss_dict[loss].to(torch.device("cpu"))), n_update)
                writer.add_scalar('val/mAP',current_mAP, n_update)

            if (i+1) % args.eval_iter == 0 :
                current_mAP = eval(model, test_loader, args.n_test, args.test_bs, device,
                                   n_update, False, writer = writer)
                model.train()

            if (i+1) % args.save_iter == 0 :
                torch.save(model.state_dict(), checkpoints_dir + '/' + str(epoch) + '.pth')

            train_loader.set_description(
                (f'epoch: {epoch}; lr: {lr:.5f}; train_loss: {loss_value:.5f}; avg_train_loss: {avg_loss:.5f}; val_mAP : {current_mAP:.5f} '
                 f'loss_objectness: {loss_objectness.item():.5f}; avg loss_rpn_box_reg: {loss_rpn_box_reg.item()*10:.5f}; '
                 f'loss_classifier: {loss_classifier.item():.3f}; avg loss_box_reg: {loss_box_reg.item():.5f}; '
                 ))
            train_loader.update(1)

        lr_scheduler.step()
