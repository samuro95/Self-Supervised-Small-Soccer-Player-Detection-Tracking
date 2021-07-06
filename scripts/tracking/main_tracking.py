import sys
import torchvision
import os
import torch
from tracking_utils import light_track
from natsort import natsorted, ns
import numpy as np


def track(
        pose_model='mobile-deconv',
        data_name='issia',
        model_name='frcnn_fpn',
        backbone='resnet18',
        checkpoint='../../checkpoints_runs/RCNN/resnet18_student.pth',
        current_model_detection=None,
        weight_loss=False,
        detection_score_thres=0.8,
        use_context=True,
        anchor_sizes=[32, 64, 128, 256, 512],
        use_track_branch_model=False,
        use_track_branch_embed=False,
        keyframe_interval=1,
        n_img_max=20,
        display_pose=False,
        frame_interval=1,
        init_frame=100,
        use_GT_position=False,
        w_spacial=0.97,
        w_visual=0.03,
        w_pose=0,
        use_IOU=True,
        use_features=True,
        use_pose=False,
        use_visual_feat=True,
        write_csv=False,
        write_video=True,
        flag_method=True,
        spacial_iou_thresh=0.5,
        visual_feat_model_name='faster-rcnn',  # or 'vgg' or 'faster-rcnn'
        imagenet_model=False,
        visualize=True,
        rescale_img_factor=0.5,
        use_filter_tracks=False,
        thres_count_ids=2,
        weight_by_score_det=False,
        use_soft_nms=True,
        nms_thres=0.4,
        N_frame_lost_keep=10,
        N_past_to_keep=1,
        N_past_to_keep_reID=3,
        max_vis_feat=4.,
        max_vis_reID=4.,
        use_ReID_module=False,
        visual_metric='l2'):

    issia_test = False

    if not use_visual_feat:
        w_visual = 0
    if not use_pose:
        w_pose = 0
    if visual_feat_model_name == 'faster-rcnn':
        imagenet_model = False

    max_dist_factor_feat = 32 * (1 / rescale_img_factor)
    max_dist_factor_reID = max_dist_factor_feat / 4

    if not use_GT_position:
        if current_model_detection is None:
            from train_tracker import get_model_detection
            model_detection = get_model_detection(model_name, weight_loss, backbone, False,
                                                  False, False, detection_score_thres, False,
                                                  use_soft_nms, anchor_sizes=anchor_sizes, use_context=use_context,
                                                  nms_thres=nms_thres, use_track_branch=use_track_branch_model)
            model_detection.load_state_dict(torch.load(checkpoint))
            model_detection.to(torch.device('cuda'))
            model_detection.eval()
        else:
            model_detection = current_model_detection
    else:
        model_detection = None

    if use_visual_feat:
        if visual_feat_model_name == 'faster-rcnn':
            if current_model_detection is None:
                from train_tracker import get_model_detection
                visual_feat_model = get_model_detection(model_name, weight_loss, backbone, False,
                                                        False, False, detection_score_thres, False,
                                                        use_soft_nms, anchor_sizes=anchor_sizes,
                                                        use_context=use_context, nms_thres=nms_thres,
                                                        use_track_branch=use_track_branch_model)
                visual_feat_model.load_state_dict(torch.load(checkpoint))
                visual_feat_model.to(torch.device('cuda'))
            else:
                visual_feat_model = current_model_detection
            visual_feat_model.eval()
            layer = visual_feat_model._modules.get('fc7')

        elif visual_feat_model_name == 'resnet50':
            visual_feat_model = torchvision.models.resnet50(pretrained=True)
            visual_feat_model.to(torch.device('cuda'))
            visual_feat_model.eval()
            layer = visual_feat_model._modules.get('avgpool')
        elif visual_feat_model_name == 'vgg19':
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

    if use_pose:
        if pose_model == 'mobile-deconv':
            from network_mobile_deconv import Network
            pose_model_path = "../other_utils/lighttrack/weights/mobile-deconv/snapshot_296.ckpt"
        elif pose_model == 'MSRA152':
            from network_MSRA152 import Network
            pose_model_path = "../other_utils/lighttrack/weights/MSRA152/MSRA_snapshot_285.ckpt"
        elif pose_model == 'CPN101':
            from network_CPN101 import Network
            pose_model_path = '../other_utils/lighttrack/weights/CPN101/CPN_snapshot_293.ckpt'
        else:
            sys.exit('pose model not available')
        # initialize pose estimator
        pose_estimator = Tester(Network(), cfg)
        pose_estimator.load_weights(pose_model_path)
    else:
        pose_estimator = None


    if data_name == 'issia':
        base_image_folder = '../../data/issia/frames/'
        base_annotation_folder = '../../data/issia/annotations/'
        rescale_bbox = [0., 0.]
    if data_name == 'SoccerNet':
        base_image_folder = '../../data/SoccerNet/sequences/'
        base_annotation_folder = None
        rescale_bbox = [0., 0.]
    if data_name == 'panorama':
        base_image_folder = '../../data/panorama/frames/'
        base_annotation_folder = None
        rescale_bbox = [0., 0.]
    if data_name == 'SPD':
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

        base_dir = os.path.join('../../data/intermediate/tracking', data_name)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        base_dir = os.path.join('../../data/intermediate/tracking', data_name, str(s))
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

        if write_csv and os.path.exists(output_csv_path):
            continue

        if issia_test:
            avg_motp = []
            avg_mota = []
            avg_idf1 = []
            acc_tab = []
            for k in range(5):
                init_frame = k * 500
                print('eval tracking on subseq', k)
                out = light_track(pose_estimator, model_detection, visual_feat_model, layer,
                                  image_folder, annotation_folder, rescale_bbox, rescale_img_factor,
                                  visualize_folder, output_video_path, output_csv_path, use_features,
                                  w_spacial, w_visual, w_pose, use_IOU, spacial_iou_thresh,
                                  detection_score_thres, use_pose, use_visual_feat, imagenet_model,
                                  display_pose, use_GT_position, flag_method, n_img_max, init_frame,
                                  frame_interval, write_csv, write_video, keyframe_interval, visualize,
                                  use_filter_tracks, thres_count_ids, weight_by_score_det, visual_metric,
                                  N_frame_lost_keep, N_past_to_keep, use_ReID_module,
                                  N_past_to_keep_reID, max_vis_feat, max_dist_factor_feat, max_vis_reID,
                                  max_dist_factor_reID,
                                  use_track_branch_embed)
                mota, motp, idf1, acc = out
                avg_motp.append(motp)
                avg_mota.append(mota)
                avg_idf1.append(idf1)
                acc_tab.append(acc)
            print('all mota ', np.mean(np.array(avg_mota)))
            print('all motp ', np.mean(np.array(avg_motp)))
            print('all idf1 ', np.mean(np.array(avg_idf1)))

            import motmetrics as mm

            mh = mm.metrics.create()
            summary = mh.compute_many(
                acc_tab,
                metrics=mm.metrics.motchallenge_metrics,
                names=['0', '1', '2', '3'],
                generate_overall=True)

            strsummary = mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names
            )
            print(strsummary)

            return (np.mean(np.array(avg_mota)), np.mean(np.array(avg_motp)), np.mean(np.array(avg_idf1)))

        else:
            out = light_track(pose_estimator, model_detection, visual_feat_model, layer,
                              image_folder, annotation_folder, rescale_bbox, rescale_img_factor,
                              visualize_folder, output_video_path, output_csv_path, use_features,
                              w_spacial, w_visual, w_pose, use_IOU, spacial_iou_thresh,
                              detection_score_thres, use_pose, use_visual_feat, imagenet_model,
                              display_pose, use_GT_position, flag_method, n_img_max, init_frame,
                              frame_interval, write_csv, write_video, keyframe_interval, visualize,
                              use_filter_tracks, thres_count_ids, weight_by_score_det, visual_metric,
                              N_frame_lost_keep, N_past_to_keep, use_ReID_module,
                              N_past_to_keep_reID, max_vis_feat, max_dist_factor_feat, max_vis_reID,
                              max_dist_factor_reID,
                              use_track_branch_embed)

            return (out)


if __name__ == '__main__':
    track()
