import torch
import numpy as np
import shutil
import cv2
import sys
import time as time
from natsort import natsorted, ns
from pascal_voc_writer import Writer
import os
import argparse
from annotation_utils import extract_annotations_for_eval, save_fields, save_player_dets, load_player_dets, \
    collect_avg_bbox_dims, segmentation_correction, initialize_pose_estimator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append('../other_utils/metrics/')
from pascalvoc import compute_metrics

sys.path.append('../detection/')
from eval_fasterRCNN import get_model_detection

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='test')
    parser.add_argument('--nb_images', type=int, default=500)
    parser.add_argument('--create_data', dest='create_data', action='store_true')
    parser.set_defaults(create_data=False)
    parser.add_argument('--model_name', type=str, default='frcnn_fpn')
    parser.add_argument('--backbone_name', type=str, default='resnet50')
    parser.add_argument('--bbox_max_h', type=int, default=200)
    parser.add_argument('--bbox_max_w', type=int, default=100)
    parser.add_argument('--bbox_intersection_thres', type=float, default=0.3)
    parser.add_argument('--bbox_score_thres', type=float, default=0.8)
    parser.add_argument('--score_thres_non_player', type=float, default=0.05)
    parser.add_argument('--no_use_field_detection', dest='use_field_detection', action='store_false')
    parser.set_defaults(use_field_detection=True)
    parser.add_argument('--field_in_thres', type=float, default=0.1)
    parser.add_argument('--seg_correct', dest='seg_correct', action='store_true')
    parser.set_defaults(seg_correct=True)
    parser.add_argument('--use_pose_estimator', dest='use_pose_estimator', action='store_true')
    parser.set_defaults(use_pose_estimator=True)
    parser.add_argument('--thres_pose_estimator', type=float, default=0.6)
    parser.add_argument('--use_SR', dest='use_SR', action='store_true')
    parser.set_defaults(use_SR=False)
    parser.add_argument('--visualization', dest='visualization', action='store_true')
    parser.set_defaults(visualization=False)
    parser.add_argument('--show_GT', dest='show_GT', action='store_true')
    parser.set_defaults(show_GT=False)
    parser.add_argument('--show_field_in', dest='show_field_in', action='store_true')
    parser.set_defaults(show_field_in=False)
    parser.add_argument('--remove_outlier_area', dest='remove_outlier_area', action='store_true')
    parser.set_defaults(remove_outlier_area=False)
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
    model_detection = get_model_detection(args.model_name, False, args.backbone_name, True,
                                          True, True, 0.05, False).to(device)
    model_detection.eval()
    data_path = '../../data/'

    if args.create_data:
        save_annotation_path = os.path.join(data_path, args.data_name, 'annotations', args.bbox_score_thres)
        if not os.path.exists(save_annotation_path):
            os.mkir(save_annotation_path)
    base_frame_path = os.path.join(data_path, args.data_name, 'frames')
    if args.data_name in ['TV_soccer', 'SPD', 'issia']:
        base_annotation_path = os.path.join(data_path, args.data_name, 'annotations')
    else:
        base_annotation_path = None
    intermediate_path = os.path.join(data_path, 'intermediate', args.data_name)
    if not os.path.exists(intermediate_path):
        os.mkdir(intermediate_path)
    if args.use_pose_estimator:
        pose_estimator = initialize_pose_estimator()

    mAP_list = []

    for s in natsorted(os.listdir(base_frame_path),
                       alg=ns.PATH | ns.IGNORECASE):  # we iterate over the sequences of the dataset

        print('sequence ', s)
        intermediate_path_seq = os.path.join(intermediate_path, s)
        if not os.path.exists(intermediate_path_seq):
            os.mkdir(intermediate_path_seq)
        frame_path = os.path.join(base_frame_path, s)
        frames = natsorted(os.listdir(frame_path), alg=ns.PATH | ns.IGNORECASE)
        frame_files = [os.path.join(frame_path, frame) for frame in frames]
        n_im = min(len(frame_files), args.nb_images)

        if args.create_data:
            save_annotation_path_seq = os.path.join(save_annotation_path, str(s))
            if not os.path.exists(save_annotation_path_seq):
                os.mkdir(save_annotation_path_seq)
            if len(os.listdir(save_annotation_path_seq)) == n_im:
                continue

        det_txt_path = os.path.join(intermediate_path_seq, 'detections')
        if os.path.exists(det_txt_path):
            shutil.rmtree(det_txt_path)
        os.mkdir(det_txt_path)

        if base_annotation_path:  # if we have GT annotation, extract them
            base_annotation_path_seq = os.path.join(base_annotation_path, s)
            gt_txt_path = os.path.join(intermediate_path_seq, 'groundtruths')
            if os.path.exists(gt_txt_path):
                shutil.rmtree(gt_txt_path)
            os.mkdir(gt_txt_path)
            image_shape = cv2.imread(frame_files[0]).shape
            extract_annotations_for_eval(args.data_name, base_annotation_path_seq, gt_txt_path, frames[:n_im],
                                         [0, 0], image_shape)

        # ==== Field detection ====

        t1 = time.time()
        field_path = os.path.join(intermediate_path_seq, 'field_detections')
        if not os.path.exists(field_path):
            os.mkdir(field_path)
        gan_path = os.path.join(intermediate_path_seq, 'gan')
        if not os.path.exists(gan_path):
            os.mkdir(gan_path)
        if len(os.listdir(field_path)) < 2 * n_im:  # if the fields of all the images are not detected
            save_fields(frame_files, True, field_path, frame_path, gan_path, n_im)
        t2 = time.time()
        print('--- field detection processed in %i sec ---' % (int(t2 - t1)))

        # ==== player detection with pretrained Faster RCNN ====

        t1 = time.time()
        intermediate_path_seq_dets = os.path.join(intermediate_path_seq, 'player_detections')
        if not os.path.exists(intermediate_path_seq_dets):
            os.mkdir(intermediate_path_seq_dets)
        det_path = os.path.join(intermediate_path_seq_dets, str(args.bbox_score_thres))
        if not os.path.exists(det_path):
            os.mkdir(det_path)
        if len(os.listdir(det_path)) < n_im:
            save_player_dets(det_path, frame_files, args.bbox_score_thres, model_detection, args.bbox_max_w,
                             args.bbox_max_h, device,n_im)
        t2 = time.time()
        print('--- player detection processed in %i sec ---' % (int(t2 - t1)))

        # ==== load images and detections ====

        images = [cv2.imread(frame_files[f]) for f in range(n_im)] # images for processing
        if args.show_field_in:
            show_images = [cv2.imread(os.path.join(field_path, str(f) + '_field_in.png')) for f in range(n_im)] # images for visualization
        else :
            show_images = [cv2.imread(frame_files[f]) for f in range(n_im)]  # images for visualization
        player_detections_list = []
        score_detection_list = []
        for f in range(n_im):
            player_det, score = load_player_dets(det_path, n_im, f, args.use_field_detection, field_path, args.field_in_thres)
            player_detections_list.append(player_det)
            score_detection_list.append(score)
        # Get avd bbox sizes and remove outliers in bbox size
        player_detections_list, avg_bbox_dims, prop_avg = collect_avg_bbox_dims(player_detections_list, images[0].shape,
                                                                                args.remove_outlier_area)
        if args.visualization:
            for f in range(n_im):
                for det in player_detections_list[f]:
                    cv2.rectangle(show_images[f],
                                  (int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])), (255, 0, 0),3)

        # ==== correction with blob detection ====

        if args.seg_correct :
            t1 = time.time()
            intermediate_path_seq_segs = os.path.join(intermediate_path_seq,'player_segmentation')
            if not os.path.exists(intermediate_path_seq_segs):
                os.mkdir(intermediate_path_seq_segs)
            player_detections_list, score_detection_list = segmentation_correction(n_im, images,
                    intermediate_path_seq_segs, player_detections_list, score_detection_list,
                    avg_bbox_dims, prop_avg, args.use_SR, args.use_pose_estimator, pose_estimator,
                    args.thres_pose_estimator, field_path, frame_files, args.visualization, show_images)
            t2 = time.time()
            print(('--- player segmentation processed in %i sec ---')%(int(t2-t1)))

        # ==== label each bbox ====

        for f in range(n_im):
            for d,det in enumerate(player_detections_list[f]) :
                score = score_detection_list[f][d]
                if score <= 1 :
                    id = 'detector'
                    #score_detection_list[f][d] = 1.
                elif score == 2 :
                    score_detection_list[f][d] = 1.
                    id = 'seg'
                else:
                    continue

        # ==== visualize ====

        if args.visualization:
            if args.show_GT:
                for f in range(n_im) :
                    path = os.path.join(gt_txt_path, frames[f][:-3]+'txt')
                    with open(path, 'r') as det_file :
                        lines = det_file.readlines()
                        for line in lines :
                            split = line.split()
                            try :
                                label, detection_score, xmin, ymin, xmax , ymax = split[0], float(split[1]), float(split[2]), float(split[3]), float(split[4]), float(split[5])
                            except :
                                label, xmin, ymin, xmax , ymax = split[0], float(split[1]), float(split[2]), float(split[3]), float(split[4])
                            det = (xmin,ymin,xmax,ymax)
                            cv2.rectangle(show_images[f], (int(det[0]), int(det[1]),int(det[2]-det[0]), int(det[3]-det[1])), (255, 255, 255), 2)
            output_path = os.path.join(intermediate_path_seq, 'output')
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            output_path = os.path.join(output_path, str(args.score_thres))
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            for f, show_image in enumerate(show_images):
                # cv2.imshow('output',show_image)
                # cv2.waitKey(10)
                cv2.imwrite(os.path.join(output_path, str(f) + '.png'), show_image)

        # ==== create annotation data ====

        if args.create_data:
            for f in range(n_im):
                image = images[f]
                image_shape = image.shape
                writer = Writer(frame_files[f], image_shape[1], image_shape[0])
                annotation_file = os.path.join(save_annotation_path_seq, str(f)+'.xml')
                for d,det in enumerate(player_detections_list[f]) :
                    score = score_detection_list[f][d]
                    if score <= 1 :
                        id = 'detector'
                    elif score == 2 :
                        id = 'seg'
                        score = 1
                    else :
                        continue
                    writer.addObject(id, det[0], det[1], det[2], det[3], difficult=score)
                writer.save(annotation_file)

        # ==== evaluate the annotation method ====

        if base_annotation_path :
            for f in range(n_im) :
                with open(os.path.join(det_txt_path,frames[f][:-4]+'.txt'),'w') as det_file :
                    for b, bbox in enumerate(player_detections_list[f]) :
                        det_file.write('person ' + str(float(score_detection_list[f][b])) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')
            mAP = compute_metrics(os.path.abspath(gt_txt_path), os.path.abspath(det_txt_path), showPlot=False)
            mAP_list.append(mAP)

    mAP_avg = np.mean(np.array(mAP_list))
    print(mAP_avg)


