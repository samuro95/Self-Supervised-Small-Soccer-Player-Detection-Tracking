import sys
import torchvision
import os
import torch
from tracking_utils import light_track
from natsort import natsorted, ns
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='issia')
    parser.add_argument('--use_GT_position', dest='use_GT_position', action='store_true')
    parser.set_defaults(use_GT_position=False)
    parser.add_argument('--rescale_img_factor', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='frcnn_fpn')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--checkpoint', type=str, default='../../checkpoints_runs/player_det_resnet18_student.pth')
    parser.add_argument('--detection_score_thres', type=float, default=0.8)
    parser.add_argument('--no_use_context', dest='use_context', action='store_false')
    parser.set_defaults(use_context=True)
    parser.add_argument('--no_use_soft_nms', dest='use_soft_nms', action='store_false')
    parser.set_defaults(use_soft_nms=True)
    parser.add_argument('--nms_thres', type=float, default=0.4)
    parser.add_argument('--anchor_sizes', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    parser.add_argument('--use_track_branch_model', dest='use_track_branch_model', action='store_true')
    parser.set_defaults(use_track_branch_model=False)
    parser.add_argument('--use_track_branch_embed', dest='use_track_branch_embed', action='store_true')
    parser.set_defaults(use_track_branch_embed=False)

    parser.add_argument('--pose_model', type=str, default='mobile-deconv')
    parser.add_argument('--keyframe_interval', type=int, default=1)
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--init_frame', type=int, default=100)
    parser.add_argument('--n_img_max', type=int, default=50)
    parser.add_argument('--no_use_IOU', dest='use_IOU', action='store_false')
    parser.set_defaults(use_IOU=True)
    parser.add_argument('--spacial_iou_thresh', type=float, default=0.5)
    parser.add_argument('--no_use_features', dest='use_features', action='store_false')
    parser.set_defaults(use_features=True)
    parser.add_argument('--no_use_visual_feat', dest='use_visual_feat', action='store_false')
    parser.set_defaults(use_visual_feat=True)
    parser.add_argument('--visual_feat_model_name', type=str, default='faster-rcnn')
    parser.add_argument('--imagenet_model', dest='imagenet_model', action='store_false')
    parser.set_defaults(imagenet_model=True)
    parser.add_argument('--use_pose', dest='use_pose', action='store_true')
    parser.set_defaults(use_pose=False)
    parser.add_argument('--weight_loss', dest='weight_loss', action='store_true')
    parser.set_defaults(weight_loss=False)
    parser.add_argument('--w_spacial', type=float, default=0.97)
    parser.add_argument('--w_visual', type=float, default=0.03)
    parser.add_argument('--w_pose', type=float, default=0.0)
    parser.add_argument('--visual_metric', type=str, default='l2')
    parser.add_argument('--use_filter_tracks', dest='use_filter_tracks', action='store_true')
    parser.set_defaults(use_filter_tracks=False)
    parser.add_argument('--thres_count_ids', type=int, default=2)

    parser.add_argument('--use_ReID_module', dest='use_ReID_module', action='store_true')
    parser.set_defaults(use_ReID_module=False)
    parser.add_argument('--max_vis_reID', type=int, default=4)
    parser.add_argument('--max_vis_feat', type=int, default=4)
    parser.add_argument('--N_past_to_keep_reID', type=int, default=3)
    parser.add_argument('--N_past_to_keep', type=int, default=1)
    parser.add_argument('--N_frame_lost_keep', type=int, default=10)

    parser.add_argument('--display_pose', dest='display_pose', action='store_true')
    parser.set_defaults(display_pose=False)
    parser.add_argument('--write_csv', dest='write_csv', action='store_true')
    parser.set_defaults(write_csv=False)
    parser.add_argument('--write_video', dest='write_video', action='store_true')
    parser.set_defaults(write_video=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--output_path', type=str, default='../../data/intermediate/tracking')

    hparams = parser.parse_args()

    hparams.current_model_detection = None
    hparams.flag_method = True
    if not hparams.use_visual_feat:
        hparams.w_visual = 0
    if not hparams.use_pose:
        hparams.w_pose = 0
    if hparams.visual_feat_model_name == 'faster-rcnn':
        hparams.imagenet_model = False

    max_dist_factor_feat = 32 * (1 / hparams.rescale_img_factor)
    max_dist_factor_reID = max_dist_factor_feat / 4

    if not hparams.use_GT_position:
        if hparams.current_model_detection is None:
            from train_tracker import get_model_detection
            model_detection = get_model_detection(hparams.model_name, hparams.weight_loss, hparams.backbone, False,
                                                  False, False, hparams.detection_score_thres, False,
                                                  hparams.use_soft_nms, anchor_sizes=hparams.anchor_sizes, use_context=hparams.use_context,
                                                  nms_thres=hparams.nms_thres, use_track_branch=hparams.use_track_branch_model)
            model_detection.load_state_dict(torch.load(hparams.checkpoint))
            model_detection.to(torch.device('cuda'))
            model_detection.eval()
        else:
            model_detection = hparams.current_model_detection
    else:
        model_detection = None

    if hparams.use_visual_feat:
        if hparams.visual_feat_model_name == 'faster-rcnn':
            if hparams.current_model_detection is None:
                from train_tracker import get_model_detection
                visual_feat_model = get_model_detection(hparams.model_name, hparams.weight_loss, hparams.backbone, False,
                                                        False, False, hparams.detection_score_thres, False,
                                                        hparams.use_soft_nms, anchor_sizes=hparams.anchor_sizes,
                                                        use_context=hparams.use_context, nms_thres=hparams.nms_thres,
                                                        use_track_branch=hparams.use_track_branch_model)
                visual_feat_model.load_state_dict(torch.load(hparams.checkpoint))
                visual_feat_model.to(torch.device('cuda'))
            else:
                visual_feat_model = hparams.current_model_detection
            visual_feat_model.eval()
            layer = visual_feat_model._modules.get('fc7')

        elif hparams.visual_feat_model_name == 'resnet50':
            visual_feat_model = torchvision.models.resnet50(pretrained=True)
            visual_feat_model.to(torch.device('cuda'))
            visual_feat_model.eval()
            layer = visual_feat_model._modules.get('avgpool')
        elif hparams.visual_feat_model_name == 'vgg19':
            visual_feat_model = torchvision.models.vgg19(pretrained=True)
            visual_feat_model.to(torch.device('cuda'))
            visual_feat_model.eval()
            layer = visual_feat_model._modules.get('avgpool')
        else:
            print(' visual feature model does not exist')
            use_visual_feat = False
    else:
        visual_feat_model = None
        layer = None

    if hparams.use_pose:
        if hparams.pose_model == 'mobile-deconv':
            from network_mobile_deconv import Network
            pose_model_path = "../other_utils/lighttrack/weights/mobile-deconv/snapshot_296.ckpt"
        elif hparams.pose_model == 'MSRA152':
            from network_MSRA152 import Network
            pose_model_path = "../other_utils/lighttrack/weights/MSRA152/MSRA_snapshot_285.ckpt"
        elif hparams.pose_model == 'CPN101':
            from network_CPN101 import Network
            pose_model_path = '../other_utils/lighttrack/weights/CPN101/CPN_snapshot_293.ckpt'
        else:
            sys.exit('pose model not available')
        # initialize pose estimator
        pose_estimator = Tester(Network(), cfg)
        pose_estimator.load_weights(pose_model_path)
    else:
        pose_estimator = None


    if hparams.data_name == 'issia':
        base_image_folder = '../../data/issia/frames/'
        base_annotation_folder = '../../data/issia/annotations/'
        rescale_bbox = [0., 0.]
    if hparams.data_name == 'SoccerNet':
        base_image_folder = '../../data/SoccerNet/sequences/'
        base_annotation_folder = None
        rescale_bbox = [0., 0.]
    if hparams.data_name == 'panorama':
        base_image_folder = '../../data/panorama/frames/'
        base_annotation_folder = None
        rescale_bbox = [0., 0.]
    if hparams.data_name == 'SPD':
        base_image_folder = '../../data/SPD/frames/'
        base_annotation_folder = None
        rescale_bbox = [0., 0.]

    for s in natsorted(os.listdir(base_image_folder), alg=ns.PATH | ns.IGNORECASE):

        print('eval tracking on seq', s)

        image_folder = base_image_folder + str(s) + '/'
        if base_annotation_folder is not None:
            annotation_folder = base_annotation_folder + str(s) + '/'
        else:
            annotation_folder = None

        base_dir = hparams.output_path + '/output_tracking'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        base_dir = os.path.join(base_dir, hparams.data_name)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        base_dir = os.path.join(base_dir, str(s))
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        visualize_folder = os.path.join(base_dir, 'visualize_tracking')
        if not os.path.exists(visualize_folder):
            os.mkdir(visualize_folder)
        output_folder = os.path.join(base_dir, 'output_tracking')
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_video_path = os.path.join(output_folder, "out.mp4")
        output_csv_path = os.path.join(output_folder, "out.csv")

        if hparams.write_csv and os.path.exists(output_csv_path):
            continue


        out = light_track(pose_estimator, model_detection, visual_feat_model, layer,
                          image_folder, annotation_folder, rescale_bbox, hparams.rescale_img_factor,
                          visualize_folder, output_video_path, output_csv_path, hparams.use_features,
                          hparams.w_spacial, hparams.w_visual, hparams.w_pose, hparams.use_IOU, hparams.spacial_iou_thresh,
                          hparams.detection_score_thres, hparams.use_pose, hparams.use_visual_feat, hparams.imagenet_model,
                          hparams.display_pose, hparams.use_GT_position, hparams.flag_method,hparams.n_img_max, hparams.init_frame,
                          hparams.frame_interval, hparams.write_csv, hparams.write_video, hparams.keyframe_interval, hparams.visualize,
                          hparams.use_filter_tracks, hparams.thres_count_ids, hparams.visual_metric,
                          hparams.N_frame_lost_keep, hparams.N_past_to_keep, hparams.use_ReID_module,
                          hparams.N_past_to_keep_reID,hparams.max_vis_feat, max_dist_factor_feat, hparams.max_vis_reID,
                          max_dist_factor_reID,
                          hparams.use_track_branch_embed)








