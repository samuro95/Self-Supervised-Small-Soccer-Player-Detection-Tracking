"""
Copyright (c) 2019 JD Finance America Corporation
"""

import sys, os, time
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import cv2
import numpy as np
import collections
import torchvision
import torchvision.transforms as T
from collections import Counter

# detector utils
import sys
sys.path.append('../other_utils/lighttrack')

# pose estimation utils
from HPE.dataset import Preprocessing
from HPE.config import cfg
sys.path.append(os.path.abspath("../other_utils/lighttrack/utils"))
sys.path.append(os.path.abspath("../other_utils/lighttrack/visualizer"))
sys.path.append(os.path.abspath("../other_utils/lighttrack/graph"))

from utils_json import *
from visualizer import *
from utils_io_file import *
from utils_io_folder import *
from natsort import natsorted, ns
import scipy.optimize as scipy_opt
import motmetrics as mm

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

import csv

flag_nms = False #Default is False, unless you know what you are doing

def show_skeleton(img,pose_keypoints_2d):
    joints = reshape_keypoints_into_joints(pose_keypoints_2d)
    img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)

def initialize_parameters():
    global video_name, img_id

    global nms_method, nms_thresh, min_scores, min_box_size
    nms_method = 'nms'
    nms_thresh = 1.
    min_scores = 1e-10
    min_box_size = 0.

    global keyframe_interval, enlarge_scale, pose_matching_threshold
    keyframe_interval = 1 # choice examples: [2, 3, 5, 8, 10, 20, 40, 100, ....]
    enlarge_scale = 0.2 # how much to enlarge the bbox before pose estimation
    pose_matching_threshold = 0.5

    global flag_flip
    flag_flip = True

    global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
    total_time_POSE = 0
    total_time_DET = 0
    total_time_ALL = 0
    total_num_FRAMES = 0
    total_num_PERSONS = 0

    global spacial_thresh
    spacial_thresh = 0.3  # The max distance between 2 frames: Default : 0.3

    global check_pose_threshold, check_pose_method
    check_pose_threshold = 0.6 # The threshold on the confidence of pose estimation.
    check_pose_method = 'max_average' # How to average the confidences of the key points for a global confidence
    return

def enlarge_bbox(bbox, scale, image_shape):
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale[0] * (max_x - min_x))
    margin_y = int(0.5 * scale[1] * (max_y - min_y))

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    min_x = max(0,min_x)
    min_y = max(0,min_y)
    max_x = min(image_shape[1],max_x)
    max_y = min(image_shape[0],max_y)

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
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

def rescale_img(img,rescale_img_factor):
    shape = img.size
    w = shape[0]
    h = shape[1]
    desired_h = h*rescale_img_factor
    desired_w = w*rescale_img_factor
    img = torchvision.transforms.Resize([int(desired_h), int(desired_w)])(img)
    w_pad = (w - desired_w)/2.
    h_pad = (h - desired_h)/2.
    img = torchvision.transforms.Pad((int(w_pad),int(h_pad)))(img)
    return(img)

def rescale_img_bbox(bbox,rescale_img_factor,image_shape):
    w = image_shape[1]
    h = image_shape[0]
    bbox = np.array(bbox)*rescale_img_factor
    target_w = w*rescale_img_factor
    target_h = h*rescale_img_factor
    w_pad = (w - target_w)/2.
    h_pad = (h - target_h)/2.
    new_w = target_w + 2*w_pad
    new_h = target_h + 2*h_pad
    bbox_center = bbox + np.array([w_pad,h_pad,w_pad,h_pad])
    return(bbox_center)

def extract_annotations(annotation_path, rescale_bbox, rescale_img_factor, image_shape):
    GT_bbox_list = []
    GT_idx_list = []
    target = parse_voc_xml(ET.parse(annotation_path).getroot())
    anno = target['annotation']
    image_path = anno['path']
    objects = anno['object']
    for obj in objects:
        idx = obj['name']
        bbox = obj['bndbox']
        bbox = [int(bbox[n]) for n in ['xmin', 'ymin', 'xmax', 'ymax']]
        bbox = enlarge_bbox(bbox,rescale_bbox,image_shape)
        bbox = rescale_img_bbox(bbox,rescale_img_factor,image_shape)
        GT_bbox_list.append(bbox)
        GT_idx_list.append(idx)
    return(GT_bbox_list, GT_idx_list,image_path)

def player_detection(image_path, rescale_img_factor, model_detection, thres_detection):
    bbox_list = []
    score_list = []
    max_w = 150
    max_h = 150
    with torch.no_grad():
        im = Image.open(image_path).convert('RGB')
        im = rescale_img(im,rescale_img_factor)
        x = [T.ToTensor()(im).to(torch.device('cuda'))]
        output, features = model_detection(x)
        output = output[0]
        scores = output['scores']
        labels =  output['labels']
        boxes = output['boxes']
        for i in range(len(scores)):
            if scores[i]>thres_detection :
                xmin,ymin,xmax,ymax = int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])
                if 0 < xmax-xmin < max_w and 0 < ymax-ymin < max_h :
                    bbox_list.append([xmin,ymin,xmax,ymax])
                    score_list.append(scores[i])
    return(bbox_list,score_list,features)


def light_track(pose_estimator, model_detection, visual_feat_model, layer,
            image_folder, annotation_folder, rescale_bbox, rescale_img_factor,
            visualize_folder, output_video_path, output_csv_path, use_features,
            w_spacial, w_visual, w_pose, use_IOU, spacial_iou_thresh, thres_detection,
            use_pose, use_visual_feat, imagenet_model,
            display_pose, use_GT_position, flag_method, n_img_max, init_frame,
            frame_interval, write_csv, write_video, keyframe_interval, visualize,
            use_filter_tracks, thres_count_ids,weight_by_score_det,visual_metric,
            N_frame_lost_keep, N_past_to_keep, use_ReID_module,
            N_past_to_keep_reID, max_vis_feat, max_dist_factor_feat, max_vis_reID, max_dist_factor_reID,
            use_track_branch):

    total_time_DET = 0
    total_num_PERSONS = 0
    total_time_ALL = 0
    total_time_POSE = 0
    total_time_FEAT = 0
    st_time_total = time.time()

    bbox_dets_list = []
    frame_prev = -1
    frame_cur = 0
    img_id = -1
    next_id = 0
    bbox_dets_list_list = []
    track_ids_dict_list = []
    GT_bbox_list_list = []
    GT_idx_list_list = []
    bbox_lost_player_list = []
    track_feat_dict_list = []

    flag_mandatory_keyframe = False

    if annotation_folder is not None :
        annotation_paths = natsorted(os.listdir(annotation_folder), alg=ns.PATH | ns.IGNORECASE)
        num_imgs = min(n_img_max, len(annotation_paths)//frame_interval - init_frame)
        total_num_FRAMES = num_imgs
    else :
        image_paths = natsorted(os.listdir(image_folder), alg=ns.PATH | ns.IGNORECASE)
        num_imgs = min(n_img_max, len(image_paths)//frame_interval - init_frame)
        total_num_FRAMES = num_imgs

    acc = mm.MOTAccumulator(auto_id=True)
    image_shape = cv2.imread(os.path.join(image_folder,os.listdir(image_folder)[0])).shape

    N_IOU = 0
    N_feat = 0
    N_reID = 0

    while img_id < num_imgs-1:

        img_id += 1

        #print("Current tracking: [image_id:{}]".format(img_id))

        if annotation_folder is not None :
            annotation_path = annotation_paths[img_id*frame_interval + init_frame]
            annotation_path = os.path.join(annotation_folder,annotation_path)
            GT_bbox_list, GT_idx_list, img_path = extract_annotations(annotation_path, rescale_bbox, rescale_img_factor, image_shape)
            GT_bbox_list_list.append(GT_bbox_list)
            GT_idx_list_list.append(GT_idx_list)
        else :
            img_path = image_paths[img_id*frame_interval + init_frame]
            img_path = os.path.join(image_folder,img_path)
        if 'u176443' in img_path :
            img_path = img_path.replace('u176443','pvitoria')

        frame_cur = img_id
        if (frame_cur == frame_prev):
            frame_prev -= 1

        if is_keyframe(img_id, keyframe_interval) or flag_mandatory_keyframe :

            flag_mandatory_keyframe = False
            bbox_dets_list = []

            # perform detection at keyframes
            st_time_detection = time.time()
            if use_GT_position :
                player_candidates = GT_bbox_list
                player_scores = torch.tensor([1.]*len(GT_bbox_list))
            else :
                #try :
                player_candidates, player_scores, img_feat = player_detection(img_path, rescale_img_factor, model_detection, thres_detection)
                # except Exception  as e :
                #     print(e)
                #     player_candidates = []
                #     player_scores = []
                #     img_feat = []
            end_time_detection = time.time()
            total_time_DET += (end_time_detection - st_time_detection)

            num_dets = len(player_candidates)
            #print("Keyframe: {} detections".format(num_dets))

            # if nothing detected at keyframe, regard next frame as keyframe because there is nothing to track
            if num_dets <= 0 :

                flag_mandatory_keyframe = True

                # add empty result
                bbox_det_dict = {"img_id": img_id,
                                 "imgpath": img_path,
                                 "det_id":  0,
                                 "track_id": None,
                                 "bbox": [0, 0, 2, 2],
                                 "visual_feat": [],
                                 "keypoints": []}
                bbox_dets_list.append(bbox_det_dict)

                bbox_dets_list_list.append(bbox_dets_list)
                track_ids_dict_list.append({})

                flag_mandatory_keyframe = True
                continue

            total_num_PERSONS += num_dets

            if img_id > 0 :   # First frame does not have previous frame
                bbox_list_prev_frame = bbox_dets_list_list[img_id - 1].copy()
                track_ids_dict_prev = track_ids_dict_list[img_id - 1].copy()

            # Perform data association

            for det_id in range(num_dets):

                # obtain bbox position
                bbox_det = player_candidates[det_id]
                score_det = float(player_scores[det_id].cpu())

                # enlarge bbox by 20% with same center position
                bbox_det = enlarge_bbox(bbox_det, [0.,0.], image_shape)

                # update current frame bbox
                bbox_det_dict = {"img_id": img_id,
                                 "det_id": det_id,
                                 "imgpath": img_path,
                                 "bbox": bbox_det,
                                 "score_det": score_det}

                if img_id == 0 or len(bbox_list_prev_frame) == 0 :   # First frame, all ids are assigned automatically
                    track_id = next_id
                    next_id += 1
                    method = None

                else : # Perform data association

                    if use_IOU :  # use IOU as first criteria
                        spacial_intersect = get_spacial_intersect(bbox_det, bbox_list_prev_frame)
                        track_id, match_index = get_track_id_SpatialConsistency(spacial_intersect, bbox_list_prev_frame, spacial_iou_thresh)
                    else :
                        track_id = -1

                    if track_id != -1:
                        method = 'spacial'
                    else :
                        method = None

                # update current frame bbox
                bbox_det_dict = {"img_id": img_id,
                                 "imgpath": img_path,
                                 "det_id": det_id,
                                 "track_id": track_id,
                                 "bbox": bbox_det,
                                 "score_det": score_det,
                                 "method": method,
                                 "visual_feat": [],
                                 "keypoints": []}

                bbox_dets_list.append(bbox_det_dict)

            # Check for repetitions in track ids and remove them.
            track_ids = [bbox_det_dict["track_id"] for bbox_det_dict in bbox_dets_list]
            track_ids_dict = collections.defaultdict(list)
            for idx, track in enumerate(track_ids) :
                track_ids_dict[track].append(idx)
            keys = list(track_ids_dict.keys())
            for track in keys :
                if len(track_ids_dict[track]) > 1 :
                    for el in track_ids_dict[track] :
                        bbox_dets_list[el]["track_id"] = -1
                        bbox_dets_list[el]["method"] = None
                    del track_ids_dict[track]

            if img_id > 0 and len(bbox_list_prev_frame) > 0 :

                # Remove already assigned elements in the previous frame.
                remaining_det_id_list = []
                prev_to_remove = []
                for det_id in range(num_dets):
                    track_id = bbox_dets_list[det_id]["track_id"]
                    if track_id == -1 :
                        remaining_det_id_list.append(det_id)
                    else :
                        prev_idx = track_ids_dict_prev[track_id]
                        prev_to_remove.append(prev_idx[0])
                        N_IOU+=1

                for index in sorted(prev_to_remove, reverse=True):
                    del bbox_list_prev_frame[index]

                # For candidates that are not associated yet

                if len(bbox_list_prev_frame) == 0 or (not use_features and not use_ReID_module) :

                    # If no more candidates in previous frame : assign new ids to remaining detections
                    for det_id in remaining_det_id_list :
                        #print('no matching')
                        bbox_dets_list[det_id]["track_id"] = next_id
                        bbox_dets_list[det_id]["method"] = None
                        track_ids_dict[next_id].append(det_id)
                        next_id += 1

                elif len(remaining_det_id_list) > 0 :

                    # For each remaining detections, perform association with a combinaison of features.
                    if (use_ReID_module or use_visual_feat) and not imagenet_model :
                        if use_GT_position :
                            img_feat, image_sizes = get_img_feat_FasterRCNN(visual_feat_model, bbox_dets_list[0]['imgpath'], rescale_img_factor)
                        img_feat_prev, image_sizes = get_img_feat_FasterRCNN(visual_feat_model, bbox_list_prev_frame[0]['imgpath'], rescale_img_factor)

                    past_track_bbox_list_list = []

                    for bbox_prev_dict in bbox_list_prev_frame :

                        prev_track_id = bbox_prev_dict['track_id']
                        past_track_idx_list = []
                        past_track_bbox_list = []
                        for i in range(1,min(N_past_to_keep,img_id)+1):
                            past_track_ids_dict = track_ids_dict_list[img_id-i]
                            if prev_track_id in past_track_ids_dict.keys() :
                                idx = past_track_ids_dict[prev_track_id][0]
                                past_track_idx_list.append(idx)
                                past_track_bbox_list.append(bbox_dets_list_list[img_id-i][idx])

                        for past_track_bbox in past_track_bbox_list :

                            if use_pose :
                                if not past_track_bbox["keypoints"] :
                                    st_time_pose = time.time()
                                    inf,_ = inference_feat_keypoints(pose_estimator, past_track_bbox)
                                    keypoints = inf[0]["keypoints"]
                                    end_time_pose = time.time()
                                    total_time_POSE += (end_time_pose - st_time_pose)
                                else :
                                    keypoints = past_track_bbox["keypoints"]
                            else :
                                keypoints = []

                            if use_visual_feat :
                                if not list(past_track_bbox["visual_feat"]) :
                                    st_time_feat = time.time()
                                    if imagenet_model :
                                        visual_feat = get_visual_feat_imagenet(visual_feat_model,layer,past_track_bbox, rescale_img_factor)
                                    else :
                                        visual_feat = get_visual_feat_fasterRCNN(visual_feat_model,past_track_bbox,img_feat_prev,image_sizes,use_track_branch)
                                    end_time_feat = time.time()
                                    total_time_FEAT += (end_time_feat - st_time_feat)
                                else :
                                    visual_feat = past_track_bbox["visual_feat"]
                            else :
                                visual_feat = []

                            past_track_bbox["keypoints"] = keypoints
                            past_track_bbox["visual_feat"] = visual_feat

                        past_track_bbox_list_list.append(past_track_bbox_list)

                    for det_id in remaining_det_id_list :

                        bbox_det_dict = bbox_dets_list[det_id]

                        if use_pose :
                            st_time_pose = time.time()
                            inf,_ = inference_feat_keypoints(pose_estimator, bbox_det_dict, rescale_img_factor)
                            keypoints = inf[0]["keypoints"]
                            end_time_pose = time.time()
                            total_time_POSE += (end_time_pose - st_time_pose)
                        else :
                            keypoints = []

                        if use_visual_feat :
                            st_time_feat = time.time()
                            if imagenet_model :
                                visual_feat = get_visual_feat_imagenet(visual_feat_model,layer,bbox_det_dict, rescale_img_factor)
                            else :
                                visual_feat = get_visual_feat_fasterRCNN(visual_feat_model,bbox_det_dict, img_feat,image_sizes,use_track_branch)
                            end_time_feat = time.time()
                            total_time_FEAT += (end_time_feat - st_time_feat)
                        else :
                            visual_feat = []

                        bbox_det_dict["keypoints"] = keypoints
                        bbox_det_dict["visual_feat"] = visual_feat

                    if use_features :

                        log = ''
                        bbox_dets_list, bbox_list_prev_frame, past_track_bbox_list_list, track_ids_dict, N_feat = feature_matching(bbox_dets_list,remaining_det_id_list, bbox_list_prev_frame,
                                                        past_track_bbox_list_list, track_ids_dict, visual_metric, max_dist_factor_feat, max_vis_feat, w_visual, w_spacial, w_pose,
                                                        use_visual_feat, use_pose, weight_by_score_det, image_shape, log, N_past_to_keep, N_feat)

                    if use_ReID_module :

                        # Adjust lost player list
                        bbox_lost_player_list = [bbox_lost_player for bbox_lost_player in bbox_lost_player_list if img_id - bbox_lost_player['img_id'] < N_frame_lost_keep]
                        bbox_lost_player_list += bbox_list_prev_frame

                        past_track_bbox_list_list_reID = []

                        for bbox_prev_dict in bbox_lost_player_list :
                            prev_track_id = bbox_prev_dict['track_id']
                            prev_im_id = bbox_prev_dict['img_id']
                            past_track_idx_list = []
                            past_track_bbox_list = []
                            for i in range(min(N_past_to_keep_reID,prev_im_id+1)):
                                past_track_ids_dict = track_ids_dict_list[prev_im_id-i]
                                if prev_track_id in past_track_ids_dict.keys() :
                                    idx = past_track_ids_dict[prev_track_id][0]
                                    past_track_idx_list.append(idx)
                                    past_track_bbox_list.append(bbox_dets_list_list[prev_im_id-i][idx])

                            for past_track_bbox in past_track_bbox_list :

                                if use_pose :
                                    if not past_track_bbox["keypoints"] :
                                        st_time_pose = time.time()
                                        inf,_ = inference_feat_keypoints(pose_estimator, past_track_bbox)
                                        keypoints = inf[0]["keypoints"]
                                        end_time_pose = time.time()
                                        total_time_POSE += (end_time_pose - st_time_pose)
                                    else :
                                        keypoints = past_track_bbox["keypoints"]
                                else :
                                    keypoints = []

                                if use_visual_feat :
                                    if not list(past_track_bbox["visual_feat"]) :
                                        st_time_feat = time.time()
                                        if imagenet_model :
                                            visual_feat = get_visual_feat_imagenet(visual_feat_model,layer,past_track_bbox, rescale_img_factor)
                                        else :
                                            visual_feat = get_visual_feat_fasterRCNN(visual_feat_model,past_track_bbox,img_feat_prev,image_sizes,use_track_branch)
                                        end_time_feat = time.time()
                                        total_time_FEAT += (end_time_feat - st_time_feat)
                                    else :
                                        visual_feat = past_track_bbox["visual_feat"]
                                else :
                                    visual_feat = []

                                past_track_bbox["keypoints"] = keypoints
                                past_track_bbox["visual_feat"] = visual_feat

                            past_track_bbox_list_list_reID.append(past_track_bbox_list)

                        #print(past_track_bbox_list_list_reID)

                        # Get non_associated dets
                        remaining_det_id_list = []
                        for det_id in range(num_dets):
                            track_id = bbox_dets_list[det_id]["track_id"]
                            if track_id == -1 :
                                remaining_det_id_list.append(det_id)

                        # Re-ID module
                        if len(remaining_det_id_list) > 0 and len(bbox_lost_player_list) > 0 :
                            log = ''
                            bbox_dets_list, bbox_lost_player_list, past_track_bbox_list_list_reID, track_ids_dict, N_reID = feature_matching(bbox_dets_list,remaining_det_id_list, bbox_lost_player_list,
                                                            past_track_bbox_list_list_reID, track_ids_dict, visual_metric, max_dist_factor_reID, max_vis_reID, w_visual, w_spacial, w_pose, use_visual_feat,
                                                            use_pose, weight_by_score_det, image_shape, log, N_past_to_keep_reID, N_reID)

                    # if still can not find a match from previous frame, then -1
                    for det_id in range(num_dets):
                        track_id = bbox_dets_list[det_id]["track_id"]
                        if track_id == -1 :
                            bbox_dets_list[det_id]["track_id"] = next_id
                            bbox_dets_list[det_id]["method"] = None
                            track_ids_dict[next_id].append(det_id)
                            next_id += 1
                else :
                    pass

            # update frame
            bbox_dets_list_list.append(bbox_dets_list)
            track_ids_dict_list.append(track_ids_dict)
            frame_prev = frame_cur

        else:
            ''' NOT KEYFRAME: multi-target pose tracking '''

            print('we only work with keyframes for now')

            # bbox_dets_list_next = []
            # keypoints_list_next = []
            #
            # num_dets = len(keypoints_list)
            # total_num_PERSONS += num_dets
            #
            # if num_dets == 0:
            #     flag_mandatory_keyframe = True
            #
            # for det_id in range(num_dets):
            #     keypoints = keypoints_list[det_id]["keypoints"]
            #
            #     # for non-keyframes, the tracked target preserves its track_id
            #     track_id = keypoints_list[det_id]["track_id"]
            #
            #     # next frame bbox
            #     bbox_det_next = get_bbox_from_keypoints(keypoints)
            #     if bbox_det_next[2] == 0 or bbox_det_next[3] == 0:
            #         bbox_det_next = [0, 0, 2, 2]
            #         total_num_PERSONS -= 1
            #     assert(bbox_det_next[2] != 0 and bbox_det_next[3] != 0) # width and height must not be zero
            #     bbox_det_dict_next = {"img_id":img_id,
            #                          "det_id":det_id,
            #                          "track_id":track_id,
            #                          "imgpath": img_path,
            #                          "bbox":bbox_det_next}
            #
            #     # next frame keypoints
            #     st_time_pose = time.time()
            #     inf_next, feat_next = inference_feat_keypoints(pose_estimator, bbox_det_dict_next)
            #     keypoints_next = inf_next[0]["keypoints"]
            #     end_time_pose = time.time()
            #     total_time_POSE += (end_time_pose - st_time_pose)
            #     #print("time for pose estimation: ", (end_time_pose - st_time_pose))
            #
            #     # check whether the target is lost
            #     target_lost = is_target_lost(keypoints_next, check_pose_method, check_pose_threshold)
            #
            #     if target_lost is False:
            #         bbox_dets_list_next.append(bbox_det_dict_next)
            #         keypoints_dict_next = {"img_id":img_id,
            #                                "det_id":det_id,
            #                                "track_id":track_id,
            #                                "imgpath": img_path,
            #                                "keypoints":keypoints_next}
            #         keypoints_list_next.append(keypoints_dict_next)
            #
            #     else:
            #         # remove this bbox, do not register its keypoints
            #         bbox_det_dict_next = {"img_id":img_id,
            #                               "det_id":  det_id,
            #                               "track_id": None,
            #                               "imgpath": img_path,
            #                               "bbox": [0, 0, 2, 2]}
            #         bbox_dets_list_next.append(bbox_det_dict_next)
            #
            #         keypoints_null = 45*[0]
            #         keypoints_dict_next = {"img_id":img_id,
            #                                "det_id":det_id,
            #                                "track_id": None,
            #                                "imgpath": img_path,
            #                                "keypoints": []}
            #         keypoints_list_next.append(keypoints_dict_next)
            #         print("Target lost. Process this frame again as keyframe. \n\n\n")
            #         flag_mandatory_keyframe = True
            #
            #         total_num_PERSONS -= 1
            #         ## Re-process this frame by treating it as a keyframe
            #         if img_id not in [0]:
            #             img_id -= 1
            #         break
            #
            # # update frame
            # if flag_mandatory_keyframe is False:
            #     bbox_dets_list = bbox_dets_list_next
            #     keypoints_list = keypoints_list_next
            #     bbox_dets_list_list.append(bbox_dets_list)
            #     keypoints_list_list.append(keypoints_list)
            #     frame_prev = frame_cur



    if use_filter_tracks :
        bbox_dets_list_list = filter_tracks(bbox_dets_list_list, thres_count_ids)

    ''' 1. statistics: get total time for lighttrack processing'''
    end_time_total = time.time()
    total_time_ALL += (end_time_total - st_time_total)

    print("N IOU : ", N_IOU)
    print("N FEAT : ", N_feat)
    print("N REID : ", N_reID)

    # visualization
    if visualize :
        print("Visualizing Tracking Results...")
        # if display_pose :
        #     show_all_from_dict(keypoints_list_list, bbox_dets_list_list, classes,
        #     joint_pairs, joint_names, image_folder, visualize_folder,
        #     display_pose = display_pose, flag_track = True, flag_method = flag_method)
        # else :
        show_all_from_dict([], bbox_dets_list_list, classes, joint_pairs, joint_names,
        rescale_img_factor = rescale_img_factor,
        img_folder_path = image_folder, output_folder_path = visualize_folder,
        display_pose = display_pose, flag_track = True, flag_method = flag_method)
        img_paths = get_immediate_childfile_paths(visualize_folder)
        if write_video:
            make_video_from_images(img_paths, output_video_path, fps=25, size=None, is_color=True, format="XVID")
            print("Visualization Finished!")
            print("Finished video {}".format(output_video_path))

            ''' Display statistics '''
            print("total_time_ALL: {:.2f}s".format(total_time_ALL))
            print("total_time_DET: {:.2f}s".format(total_time_DET))
            print("total_time_POSE: {:.2f}s".format(total_time_POSE))
            print("total_time_FEAT: {:.2f}s".format(total_time_FEAT))
            print("total_time_TRACK: {:.2f}s".format(total_time_ALL - total_time_DET - total_time_POSE - total_time_FEAT))
            print("total_num_FRAMES: {:d}".format(total_num_FRAMES))
            print("total_num_PERSONS: {:d}\n".format(total_num_PERSONS))
            print("Average FPS: {:.2f}fps".format(total_num_FRAMES / total_time_ALL))
            print("Average FPS for Detection only : {:.2f}fps".format(total_num_FRAMES / (total_time_DET)))
            print("Average FPS excluding Detection: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_DET)))
            print("Average FPS for framework only: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_DET - total_time_POSE - total_time_FEAT) ))

    if write_csv is True :
        print(output_csv_path)
        write_tracking_csv(bbox_dets_list_list, output_csv_path)
        print("total_time_ALL: {:.2f}s".format(total_time_ALL))

    # compute metrics
    if annotation_folder is not None :
        try :
            mota,mptp,idf1,acc = evaluate_tracking(bbox_dets_list_list, GT_idx_list_list, GT_bbox_list_list)
            return(mota,mptp,idf1,acc)
        except Exception as e :
            print(e)
            print('no evaluation worked')
            return(0,0,0)


# def get_track_id_SGCN(bbox_cur_frame, bbox_list_prev_frame, keypoints_cur_frame, keypoints_list_prev_frame):
#     assert(len(bbox_list_prev_frame) == len(keypoints_list_prev_frame))
#
#     min_index = None
#     min_matching_score = sys.maxsize
#     global pose_matching_threshold
#     # if track_id is still not assigned, the person is really missing or track is really lost
#     track_id = -1
#
#     for det_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
#         bbox_prev_frame = bbox_det_dict["bbox"]
#
#         # check the pose matching score
#         keypoints_dict = keypoints_list_prev_frame[det_index]
#         keypoints_prev_frame = keypoints_dict["keypoints"]
#
#         pose_matching_score = get_pose_matching_score(keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame)
#
#         if pose_matching_score <= pose_matching_threshold and pose_matching_score <= min_matching_score:
#             # match the target based on the pose matching score
#             min_matching_score = pose_matching_score
#             min_index = det_index
#
#     if min_index is None:
#         return -1, None
#     else:
#         print('matching with GCN')
#         track_id = bbox_list_prev_frame[min_index]["track_id"]
#         return track_id, min_index

def feature_matching(bbox_dets_list, remaining_det_id_list, bbox_list_prev_frame, past_track_bbox_list_list, track_ids_dict,
        visual_metric, max_dist_factor, max_vis, w_visual, w_spacial, w_pose, use_visual_feat,
        use_pose, weight_by_score_det, image_shape, log, N_past_to_keep, N_meth, show_track = False, show_NN = False):

    dist_tab = []
    weight_tab = []
    spacial_dist = np.array([list(get_spacial_distance(bbox_dets_list[det_id]["bbox"], past_track_bbox_list_list, image_shape)) for det_id in remaining_det_id_list])
    dist_tab.append(spacial_dist)
    weight_tab.append(w_spacial)
    if use_visual_feat :
        visual_dist = np.array([list(get_visual_similarity(bbox_dets_list[det_id]['visual_feat'], past_track_bbox_list_list, N_past_to_keep, metric = visual_metric)) for det_id in remaining_det_id_list])
        dist_tab.append(visual_dist)
        weight_tab.append(w_visual)
    if use_pose :
        pose_dist = np.array([list(1-get_pose_similarity(bbox_dets_list[det_id]["bbox"],past_track_bbox_list_list, bbox_dets_list[idx]["keypoints"])) for idx,det_id in enumerate(remaining_det_id_list)])
        dist_tab.append(pose_dist)
        weight_tab.append(w_pose)

    # if weight_by_score_det :
    #     weight_tab = bbox_dets_list[det_id]["score_det"]*np.array(weight_tab)

    # for 5 players : display first visual similarity players for control
    if show_track :
        for i in range(5):
            bbox_det = bbox_dets_list[-i]
            img_path = bbox_det['imgpath']
            img = Image.open(img_path).convert('RGB')
            img = rescale_img(img,0.6)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bbox = bbox_det['bbox']
            patch = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            cv2.imshow('ref',patch)
            for j in range(len(past_track_bbox_list_list[i])) :
                bbox_past_frame = past_track_bbox_list_list[i][j]
                img_path = bbox_past_frame['imgpath']
                img = Image.open(img_path).convert('RGB')
                img = rescale_img(img,0.6)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bbox = bbox_past_frame['bbox']
                patch = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imsave(str(id)+'.png',patch)

    if show_NN :
        for i in range(3):
            det_id = remaining_det_id_list[i]
            bbox_curr_frame = bbox_dets_list[det_id]
            img_path = bbox_curr_frame['imgpath']
            img = Image.open(img_path).convert('RGB')
            img = rescale_img(img,0.6)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bbox = bbox_curr_frame['bbox']
            patch = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            best_visual_similarities = np.argsort(visual_dist[i])
            cv2.imshow('ref',patch)
            for id,j in enumerate(best_visual_similarities[:2]) :
                past_track = past_track_bbox_list_list[j]
                bbox_prev_frame = past_track[0]
                img_path = bbox_prev_frame['imgpath']
                img = Image.open(img_path).convert('RGB')
                img = rescale_img(img,0.6)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bbox = bbox_prev_frame['bbox']
                patch = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imsave(str(i)+'_'+str(id)+'.png',patch)

    # _, axs = plt.subplots(5, 5, figsize=(12, 12))
    # axs = axs.flatten()
    # for img, ax in zip(imgs, axs):
    #     ax.imshow(img)
    #     plt.show()

    distx = image_shape[0]/max_dist_factor
    disty = image_shape[1]/max_dist_factor
    max_dist = np.sqrt(distx**2+disty**2)/np.sqrt(image_shape[0]**2 + image_shape[1]**2)
    matches = compute_matches(dist_tab, weight_tab, max_dist = max_dist, max_vis = max_vis, bipart_match_algo = 'hungarian')

    idx_to_remove_prev = []
    for i,match in enumerate(matches):
        track_id = bbox_list_prev_frame[match]["track_id"]
        if match != -1:
            det_id = remaining_det_id_list[i]
            #print('matching with feature matrix')
            bbox_dets_list[det_id]["track_id"] = track_id
            bbox_dets_list[det_id]["method"] = log
            track_ids_dict[track_id].append(det_id)
            idx_to_remove_prev.append(match)
            N_meth += 1
            #print(log)

        # if still can not find a match from previous frame, then -1

        if match == -1 :
            #print('no matching with feature matrix')
            det_id = remaining_det_id_list[i]
            bbox_dets_list[det_id]["track_id"] = -1
            bbox_dets_list[det_id]["method"] = None
            #bbox_dets_list[det_id]["track_id"] = next_id
            #bbox_dets_list[det_id]["method"] = None
            #track_ids_dict[next_id].append(det_id)
            #next_id += 1

    for index in sorted(idx_to_remove_prev, reverse=True):
        del past_track_bbox_list_list[index]
        del bbox_list_prev_frame[index]

    return(bbox_dets_list, bbox_list_prev_frame, past_track_bbox_list_list, track_ids_dict, N_meth)

def evaluate_tracking(bbox_dets_list_list, GT_idx_list_list, GT_bbox_list_list) :

    acc = mm.MOTAccumulator(auto_id=True)
    for f, bbox_dets_list in enumerate(bbox_dets_list_list) :
        track_ids = [el['track_id'] for el in bbox_dets_list]
        track_boxes = [el['bbox'] for el in bbox_dets_list]
        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])

        gt_ids = GT_idx_list_list[f]
        gt_boxes = GT_bbox_list_list[f]
        if gt_ids :
            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
        acc.update(gt_ids, track_ids, distance)

    # mh = mm.metrics.create()
    # summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
    # summary1 = mh.compute(acc, metrics=['num_unique_objects','num_detections','precision','recall'], name='acc1')
    # print(summary)
    # print(summary1)

    mh = mm.metrics.create()

    summary = mh.compute_many(
    [acc],
    metrics=mm.metrics.motchallenge_metrics,
    names=['full'])

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    print(strsummary)

    out_score = mh.compute(acc, metrics=['mota', 'motp','idf1'], name='acc', return_dataframe=False)

    mota = out_score['mota']
    motp = out_score['motp']
    idf1 = out_score['idf1']

    return(mota,motp,idf1,acc)


def filter_tracks(bbox_dets_list_list, thres_count_ids = 1):
    all_track_ids = [bbox_det['track_id'] for bbox_dets_list in bbox_dets_list_list for bbox_det in bbox_dets_list]
    ids_counter = Counter(all_track_ids)
    track_ids_to_remove = []
    n = 0
    for k,v in ids_counter.items() :
        if v <= thres_count_ids :
            track_ids_to_remove.append(k)
            n+=1
    print(n, 'tracks removed out of ', len(ids_counter.keys()))
    for b,bbox_dets_list in enumerate(bbox_dets_list_list) :
        remlist = []
        for bb,bbox_det in enumerate(bbox_dets_list) :
            track_id = bbox_det['track_id']
            if track_id in track_ids_to_remove :
                remlist.append(bb)
        for index in sorted(remlist, reverse=True):
            del bbox_dets_list_list[b][index]
    return(bbox_dets_list_list)

def get_pose_similarity(bbox_cur_frame, bbox_list_prev_frame, keypoints_cur_frame, keypoints_list_prev_frame):

    pose_sim = np.zeros(len(bbox_list_prev_frame))

    for det_index, bbox_det_dict in enumerate(bbox_list_prev_frame):

        bbox_prev_frame = bbox_det_dict["bbox"]
        keypoints_dict = keypoints_list_prev_frame[det_index]
        keypoints_prev_frame = keypoints_dict["keypoints"]
        pose_matching_score = get_pose_matching_score(keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame)
        pose_sim[det_index] = pose_matching_score

    return(pose_sim)


def get_track_id_SpatialConsistency(spacial_similarities, bbox_list_prev_frame, spacial_thresh):

    if len(spacial_similarities) == 1 :
        if spacial_similarities[0] > spacial_thresh :
            max_index = 0
            track_id = bbox_list_prev_frame[max_index]["track_id"]
            #print('matching with dist IOU :', spacial_similarities[0])
            return track_id, max_index
        else :
            return -1, None

    sim_argsort = np.argsort(spacial_similarities)
    sim_sort = spacial_similarities[sim_argsort]

    if sim_sort[-1] <= 0 :
        return -1, None
    elif sim_sort[-1] > 0 and sim_sort[-2] <= 0 :
        max_index = sim_argsort[-1]
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        #print('matching with dist IOU :', sim_sort[-1])
        return track_id, max_index
    else :
        if sim_sort[-1]>0.5*sim_sort[-2] and sim_sort[-1] > spacial_thresh :
            max_index = sim_argsort[-1]
            track_id = bbox_list_prev_frame[max_index]["track_id"]
            #print('matching with dist IOU :', sim_sort[-1])
            return track_id, max_index
        else :
            return -1, None

def get_spacial_intersect(bbox_cur_frame, bbox_list_prev_frame):

    spacial_sim = np.zeros(len(bbox_list_prev_frame))

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]
        boxA = bbox_cur_frame
        boxB = bbox_prev_frame
        spacial_sim[bbox_index] = iou(boxA, boxB)

    return(spacial_sim)

def get_spacial_distance(bbox_cur_frame, past_track_bbox_list_list, image_shape):

    bbox_list_prev_frame = [past_track_bbox_list[0] for past_track_bbox_list in past_track_bbox_list_list]

    spacial_sim = np.zeros(len(bbox_list_prev_frame))

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]
        centAx = (bbox_cur_frame[0]+bbox_cur_frame[2])/2.
        centAy = (bbox_cur_frame[1]+bbox_cur_frame[3])/2.
        centBx = (bbox_prev_frame[0]+bbox_prev_frame[2])/2.
        centBy = (bbox_cur_frame[1]+bbox_cur_frame[3])/2.
        distx = np.abs(centAx-centBx)
        disty = np.abs(centAy-centBy)
        dist = np.sqrt(distx**2+disty**2)/np.sqrt(image_shape[0]**2 + image_shape[1]**2)
        spacial_sim[bbox_index] = dist

    return(spacial_sim)

def get_visual_similarity(feat, past_track_bbox_list_list, N_past_to_keep, metric = 'cos_similarity') :
    weights = np.array([(1/2)**n for n in range(N_past_to_keep)])
    weights = np.array([(1)**n for n in range(N_past_to_keep)])
    res = []
    feat = np.array(feat)
    for past_track_bbox_list in past_track_bbox_list_list :
        feat_vector = np.array([past_track_bbox_list[i]["visual_feat"].numpy()*weights[i] for i in range(len(past_track_bbox_list))])
        feat_vector = np.mean(feat_vector,axis=0)
        if metric == 'cos_similarity' :
            res.append(np.dot(feat/np.linalg.norm(feat),feat_vector/np.linalg.norm(feat_vector)))
        if metric == 'correlation' :
            res.append(np.dot(feat,feat_vector))
        if metric == 'l1' :
            res.append(np.linalg.norm(feat-feat_vector,1))
        if metric == 'l2' :
            res.append(np.linalg.norm(feat-feat_vector,2))
    return(np.array(res))

def get_pose_matching_score(keypoints_A, keypoints_B, bbox_A, bbox_B):
    if keypoints_A == [] or keypoints_B == []:
        print("a graph not correctly generated!")
        return sys.maxsize

    graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
    if flag_pass_check is False:
        print("c graph not correctly generated!")
        return sys.maxsize

    graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
    if flag_pass_check is False:
        print("d graph not correctly generated!")
        return sys.maxsize

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    start = time.time()
    flag_match, dist = pose_matching(data_A, data_B)
    end = time.time()
    return dist


def get_iou_score(bbox_gt, bbox_det):
    iou_score = iou(boxA, boxB)
    #print("iou_score: ", iou_score)
    return iou_score


def is_target_lost(keypoints, method, check_pose_threshold):
    num_keypoints = int(len(keypoints) / 3.0)
    if method == "average":
        # pure average
        score = 0
        for i in range(num_keypoints):
            score += keypoints[3*i + 2]
        score /= num_keypoints*1.0
        print("target_score: {}".format(score))
    elif method == "max_average":
        score_list = keypoints[2::3]
        score_list_sorted = sorted(score_list)
        top_N = 4
        assert(top_N < num_keypoints)
        top_scores = [score_list_sorted[-i] for i in range(1, top_N+1)]
        score = sum(top_scores)/top_N
    if score < check_pose_threshold :
        return True
    else:
        return False


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox_from_keypoints(keypoints_python_data, img_shape):
    if keypoints_python_data == [] or keypoints_python_data == 45*[0]:
        return [0, 0, 2, 2]

    num_keypoints = len(keypoints_python_data)
    x_list = []
    y_list = []
    for keypoint_id in range(int(num_keypoints / 3)):
        x = keypoints_python_data[3 * keypoint_id]
        y = keypoints_python_data[3 * keypoint_id + 1]
        vis = keypoints_python_data[3 * keypoint_id + 2]
        if vis != 0 and vis!= 3:
            x_list.append(x)
            y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    scale = enlarge_scale # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], [scale,scale], img_shape)
    bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return bbox_in_xywh



def get_visual_feat_imagenet(model,layer,data, rescale_img_factor):
    with torch.no_grad():
        scaler = transforms.Scale((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        img = Image.open(data['imgpath']).convert('RGB')
        img = rescale_img(img,rescale_img_factor)
        bbox = data['bbox']
        box = (bbox[0],bbox[1],bbox[2],bbox[3])
        patch = img.crop(box)
        t_img = Variable(normalize(to_tensor(scaler(patch))).unsqueeze(0)).to(torch.device('cuda'))
        my_embedding = torch.zeros(2048)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.squeeze())
        h = layer.register_forward_hook(copy_data)
        model(t_img)
        h.remove()
        feat = my_embedding
    return feat

def get_img_feat_FasterRCNN(model,img_path,rescale_img_factor):
    with torch.no_grad():
        image = Image.open(img_path).convert('RGB')
        image = rescale_img(image,rescale_img_factor)
        image = [T.ToTensor()(image).to(torch.device('cuda'))]
        image,_ = model.transform(image, None)
        features = model.backbone(image.tensors)
        return(features,image.image_sizes)

def get_visual_feat_fasterRCNN(model,data,features,image_sizes,use_track_branch):
    with torch.no_grad():
        bbox = data['bbox']
        box = (float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3]))
        proposals = [torch.tensor([box]).to(torch.device('cuda'))]
        if not use_track_branch :
            feat = model.roi_heads(features, proposals, image_sizes, get_feature_only=True)[0].cpu()
        else :
            feat = model.track_heads(features, proposals, image_sizes)[0].cpu()
            feat2 = model.roi_heads(features, proposals, image_sizes, get_feature_only=True)[0].cpu()
        return feat

def inference_feat_keypoints(pose_estimator, test_data, flag_nms=False):
    cls_dets = test_data["bbox"]
    # nms on the bboxes
    if flag_nms is True:
        cls_dets, keep = apply_nms(cls_dets, nms_method, nms_thresh)
        test_data = np.asarray(test_data)[keep]
        if len(keep) == 0:
            return -1
    else:
        test_data = [test_data]

    # crop and detect pose
    pose_heatmaps, feat, details, cls_skeleton, crops, start_id, end_id = get_pose_feat_from_bbox(pose_estimator, test_data, cfg)
    # get keypoint positions from pose
    keypoints = get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id)
    # dump results
    pose_results = prepare_results(test_data[0], keypoints, cls_dets)
    #feat /= np.linalg.norm(feat)
    return pose_results, feat


def apply_nms(cls_dets, nms_method, nms_thresh):
    # nms and filter
    keep = np.where((cls_dets[:, 4] >= min_scores) &
                    ((cls_dets[:, 3] - cls_dets[:, 1]) * (cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size))[0]
    cls_dets = cls_dets[keep]
    if len(cls_dets) > 0:
        if nms_method == 'nms':
            keep = gpu_nms(cls_dets, nms_thresh)
        elif nms_method == 'soft':
            keep = cpu_soft_nms(np.ascontiguousarray(cls_dets, dtype=np.float32), method=2)
        else:
            assert False
    cls_dets = cls_dets[keep]
    return cls_dets, keep


def get_pose_feat_from_bbox(pose_estimator, test_data, cfg):
    cls_skeleton = np.zeros((len(test_data), cfg.nr_skeleton, 3))
    crops = np.zeros((len(test_data), 4))

    batch_size = 1
    start_id = 0
    end_id = min(len(test_data), batch_size)

    test_imgs = []
    details = []
    for i in range(start_id, end_id):
        test_img, detail = Preprocessing(test_data[i], stage='test')
        test_imgs.append(test_img)
        details.append(detail)

    details = np.asarray(details)
    feed = test_imgs
    for i in range(end_id - start_id):
        ori_img = test_imgs[i][0].transpose(1, 2, 0)
        if flag_flip == True:
            flip_img = cv2.flip(ori_img, 1)
            feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])
    feed = np.vstack(feed)

    predict = pose_estimator.predict_one([feed.transpose(0, 2, 3, 1).astype(np.float32)])
    res = predict[0]
    res = res.transpose(0, 3, 1, 2)

    try :
        feat = predict[1].squeeze()[0,:]
        feat /= np.linalg.norm(feat) # 2 x 1024
    except :
        feat = None

    if flag_flip == True:
        for i in range(end_id - start_id):
            fmp = res[end_id - start_id + i].transpose((1, 2, 0))
            fmp = cv2.flip(fmp, 1)
            fmp = list(fmp.transpose((2, 0, 1)))
            for (q, w) in cfg.symmetry:
                fmp[q], fmp[w] = fmp[w], fmp[q]
            fmp = np.array(fmp)
            res[i] += fmp
            res[i] /= 2

    pose_heatmaps = res

    return pose_heatmaps, feat,  details, cls_skeleton, crops, start_id, end_id


def get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id):
    res = pose_heatmaps

    for test_image_id in range(start_id, end_id):

        r0 = res[test_image_id - start_id].copy()
        r0 /= 255.
        r0 += 0.5

        for w in range(cfg.nr_skeleton):
            res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])

        border = 10
        dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
        dr[:, border:-border, border:-border] = res[test_image_id - start_id][:cfg.nr_skeleton].copy()

        for w in range(cfg.nr_skeleton):
            dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)

        for w in range(cfg.nr_skeleton):
            lb = dr[w].argmax()
            y, x = np.unravel_index(lb, dr[w].shape)
            dr[w, y, x] = 0
            lb = dr[w].argmax()
            py, px = np.unravel_index(lb, dr[w].shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, cfg.output_shape[1] - 1))
            y = max(0, min(y, cfg.output_shape[0] - 1))
            cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)
            cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

        # map back to original images
        crops[test_image_id, :] = details[test_image_id - start_id, :]
        for w in range(cfg.nr_skeleton):
            cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / cfg.data_shape[1] * (crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
            cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / cfg.data_shape[0] * (crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]

    return cls_skeleton


def prepare_results(test_data, cls_skeleton, cls_dets):
    cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, cfg.nr_skeleton)

    cls_scores = 1
    dump_results = []
    cls_skeleton = np.concatenate(
        [cls_skeleton.reshape(-1, cfg.nr_skeleton * 3), (cls_scores * cls_partsco.mean(axis=1))[:, np.newaxis]],
        axis=1)
    for i in range(len(cls_skeleton)):
        result = dict(image_id=test_data['img_id'],
                      category_id=1,
                      score=float(round(cls_skeleton[i][-1], 4)),
                      keypoints=cls_skeleton[i][:-1].round(3).tolist())
        dump_results.append(result)
    return dump_results

def is_keyframe(img_id, interval=10):
    if img_id % interval == 0:
        return True
    else:
        return False

def make_my_json(nframe, dets_list_list,output_file):
    final = {}
    final['frames'] = []
    for img_id in range(nframe):
        current_dict = {}
        current_dict["img_id"] = img_id
        current_dict["class"] = "frame"
        current_dict["hypotheses"] = []
        final['frames'].append(current_dict)
    final['class'] = "video"
    final['filename'] = "file.idx"

    for i in range(len(dets_list_list)):
        dets_list = dets_list_list[i]
        if dets_list == []:
            continue
        for j in range(len(dets_list)):
            bbox = dets_dict["bbox"][0:4]
            img_id = dets_dict["img_id"]
            track_id = dets_dict["track_id"]
            current_ann = {"id": track_id, "x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}
            final['frames'][img_id]["hypotheses"].append(current_ann)
    return(final)

def write_tracking_csv(bbox_dets_list_list, output_file):
    with open(output_file, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for frame_id,bbox_dets_list in enumerate(bbox_dets_list_list) :
            for bbox_dets in bbox_dets_list :
                bbox = bbox_dets["bbox"][0:4]
                to_write = [bbox_dets["img_id"],bbox_dets["track_id"], bbox[0], bbox[1], bbox[2], bbox[3], -1, -1, -1, -1]
                writer.writerow(to_write)


# def pose_to_standard_mot(keypoints_list_list, dets_list_list):
#     openSVAI_python_data_list = []
#
#     num_keypoints_list = len(keypoints_list_list)
#     num_dets_list = len(dets_list_list)
#     assert(num_keypoints_list == num_dets_list)
#
#     for i in range(num_dets_list):
#
#         dets_list = dets_list_list[i]
#         keypoints_list = keypoints_list_list[i]
#
#         if dets_list == []:
#             continue
#         img_path = dets_list[0]["imgpath"]
#         img_folder_path = os.path.dirname(img_path)
#         img_name =  os.path.basename(img_path)
#         img_info = {"folder": img_folder_path,
#                     "name": img_name,
#                     "id": [int(i)]}
#         openSVAI_python_data = {"image":[], "candidates":[]}
#         openSVAI_python_data["image"] = img_info
#
#         num_dets = len(dets_list)
#         num_keypoints = len(keypoints_list) #number of persons, not number of keypoints for each person
#         candidate_list = []
#
#         for j in range(num_dets):
#             keypoints_dict = keypoints_list[j]
#             dets_dict = dets_list[j]
#
#             img_id = keypoints_dict["img_id"]
#             det_id = keypoints_dict["det_id"]
#             track_id = keypoints_dict["track_id"]
#             img_path = keypoints_dict["imgpath"]
#
#             bbox_dets_data = dets_list[det_id]
#             det = dets_dict["bbox"]
#             if  det == [0, 0, 2, 2]:
#                 # do not provide keypoints
#                 candidate = {"det_bbox": [0, 0, 2, 2],
#                              "det_score": 0}
#             else:
#                 bbox_in_xywh = det[0:4]
#                 keypoints = keypoints_dict["keypoints"]
#
#                 track_score = sum(keypoints[2::3])/len(keypoints)/3.0
#
#                 candidate = {"det_bbox": bbox_in_xywh,
#                              "det_score": 1,
#                              "track_id": track_id,
#                              "track_score": track_score,
#                              "pose_keypoints_2d": keypoints}
#             candidate_list.append(candidate)
#         openSVAI_python_data["candidates"] = candidate_list
#         openSVAI_python_data_list.append(openSVAI_python_data)
#     return openSVAI_python_data_list


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]

def bbox_valid(bbox,image_shape):
    valid = 0<=bbox[0]<=image_shape[1] and 0<=bbox[2]<=image_shape[1] and 0<=bbox[1]<=image_shape[0] and 0<=bbox[3]<=image_shape[0]
    if bbox == [0, 0, 2, 2]:
        valid=False
    return valid

def filter_detections(human_candidates, image_shape):
    res = []
    for det in human_candidates :
        if bbox_valid(det, image_shape) :
            res.append(det)
    return(det)


def bipartite_matching_greedy(C):
    """
    Code from https://github.com/facebookresearch/DetectAndTrack/blob/master/lib/core/tracking_engine.py
    Computes the bipartite matching between the rows and columns, given the
    cost matrix, C.
    """
    C = C.copy()  # to avoid affecting the original matrix
    prev_ids = []
    cur_ids = []
    row_ids = np.arange(C.shape[0])
    col_ids = np.arange(C.shape[1])
    while C.size > 0:
        # Find the lowest cost element
        i, j = np.unravel_index(C.argmin(), C.shape)
        # Add to results and remove from the cost matrix
        row_id = row_ids[i]
        col_id = col_ids[j]
        prev_ids.append(row_id)
        cur_ids.append(col_id)
        C = np.delete(C, i, 0)
        C = np.delete(C, j, 1)
        row_ids = np.delete(row_ids, i, 0)
        col_ids = np.delete(col_ids, j, 0)
    return prev_ids, cur_ids


def compute_matches(similarity_tab, weight_tab, max_dist = 100., max_vis = 100., bipart_match_algo = 'hungarian'):

    # matches structure keeps track of which of the current boxes matches to
    # which box in the previous frame. If any idx remains -1, it will be set
    # as a new track.

    C = np.average(np.array(similarity_tab), axis = 0, weights=weight_tab).transpose()
    C_dist = np.array(similarity_tab[0]).transpose()
    C_vis = np.array(similarity_tab[1]).transpose()

    matches = -np.ones((C.shape[1],), dtype=np.int32)

    if bipart_match_algo == 'hungarian':
        prev_inds, next_inds = scipy_opt.linear_sum_assignment(C)
    elif bipart_match_algo == 'greedy':
        prev_inds, next_inds = bipartite_matching_greedy(C)
    else:
        raise NotImplementedError('Unknown matching algo: {}'.format(
            bipart_match_algo))
    assert(len(prev_inds) == len(next_inds))

    for i in range(len(prev_inds)):
        cost = C[prev_inds[i], next_inds[i]]
        dist = C_dist[prev_inds[i], next_inds[i]]
        vis = C_vis[prev_inds[i], next_inds[i]]
        if dist < max_dist and vis < max_vis :
            matches[next_inds[i]] = prev_inds[i]
        else :
            matches[next_inds[i]] = -1
    return matches
