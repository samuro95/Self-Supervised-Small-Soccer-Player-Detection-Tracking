import os
import sys
import collections

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import subprocess
from natsort import natsorted, ns
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision

sys.path.append('../other_utils/lighttrack/')
sys.path.append('../other_utils/lighttrack/utils')
sys.path.append('../other_utils/lighttrack/visualizer')
sys.path.append('../other_utils/lighttrack/graph')
from utils_json import *
from visualizer import *
from utils_io_file import *
from utils_io_folder import *
from HPE.config import cfg
from tfflat.base import Tester
from tracking_utils import initialize_parameters, get_bbox_from_keypoints, inference_feat_keypoints
from network_CPN101 import Network

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


def enlargeBbox(bbox, scale, image_shape):
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale[0] * (max_x - min_x))
    margin_y = int(0.5 * scale[1] * (max_y - min_y))

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image_shape[1], max_x)
    max_y = min(image_shape[0], max_y)

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged

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

def extract_annotations_for_eval(data_name, base_annotation_path, gt_txt_path, frames, rescale_bbox, image_shape):
    for f, fname in enumerate(frames):
        GT_bbox_list = []
        GT_idx_list = []
        if data_name == 'TV_soccer':
            annotation_file = os.path.join(base_annotation_path, fname[:-3] + 'xml')
        elif data_name == 'barca':
            annotation_file = os.path.join(base_annotation_path, fname[:-3] + 'xml')
        elif data_name == 'eval':
            annotation_file = os.path.join(base_annotation_path, str(frames[f])[:-3] + 'xml')
        else:
            annotation_file = os.path.join(base_annotation_path, fname[:-3] + 'xml')

        with open(os.path.join(gt_txt_path, frames[f][:-3] + 'txt'), 'w') as GT_file:

            target = parse_voc_xml(ET.parse(annotation_file).getroot())
            anno = target['annotation']
            objects = anno['object']
            for obj in objects:
                idx = obj['name']
                bbox = obj['bndbox']
                if not False in [el in bbox.keys() for el in ['xmin', 'ymin', 'xmax', 'ymax']]:
                    bbox = [int(float(bbox[n])) for n in ['xmin', 'ymin', 'xmax', 'ymax']]
                    bbox = enlargeBbox(bbox, rescale_bbox, image_shape)
                    GT_bbox_list.append(bbox)
                    GT_idx_list.append(idx)
                    GT_file.write(
                        'person ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')


def field_selection(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image_shape = img.shape
    lower_green = np.array([15, 50, 50])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT)

    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 1
    scd_max_area = 0
    max_index = 0
    scd_max_index = 0

    for c, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            scd_max_area = max_area
            scd_max_index = max_index
            max_area = area
            max_index = c
        elif area > scd_max_area:
            scd_max_area = area
            scd_max_index = c

    cnt = contours[max_index]
    hull = cv2.convexHull(cnt)
    field_mask = np.zeros((image_shape[0], image_shape[1]))
    cv2.drawContours(field_mask, [hull], 0, 255, -1)

    if scd_max_area > 0.1 * max_area:
        cnt = contours[scd_max_index]
        hull = cv2.convexHull(cnt)
        cv2.drawContours(field_mask, [hull], 0, 255, -1)

    field_in = np.zeros_like(img)
    field_in[field_mask == 255, :] = img[field_mask == 255, :]

    field_out = np.ones_like(img) * 255
    field_out[field_mask == 0, :] = img[field_mask == 0, :]

    return field_in, field_mask


def select_mean_bottom_value(im,xmin,xmax,y):
    sum = 0
    count = 0
    for i in range(xmin,xmax):
        for j in range(int(y(i)),im.shape[0]):
            sum += im[j,i]/255
            count += 1
    return sum/count


def color_bottom_line(field_line_gan, field_mask_gan):
    lines = cv2.HoughLines(field_line_gan, 2, 3 * np.pi / 180, 700)
    image_shape = field_mask_gan.shape
    d = max(image_shape[0], image_shape[1])
    if lines is not None and len(lines) > 0:
        for l, line in enumerate(lines):
            r, t = line[0]
            a, b = np.cos(t), np.sin(t)
            x0, y0 = a * r, b * r
            if a == 0:
                x1 = x2 = x0
                y1 = 0
                y2 = image_shape[0]
                y = None
            elif b == 0:
                y1 = y2 = y0
                x1 = 0
                x2 = image_shape[1]
                y = lambda x: y0
            else:
                m = -a / b
                d = -m * x0 + y0
                y = lambda x: m * x + d
                x1, x2 = 0, image_shape[1]
                y1, y2 = y(x1), y(x2)
                if y1 < 0:
                    y1 = 0
                    x1 = -d / m
                if y2 > image_shape[0]:
                    y2 = image_shape[0]
                    x2 = (image_shape[0] - d) / m
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if y is not None and min(y1, y2) > image_shape[0] / 2:
                mean_bottom = select_mean_bottom_value(field_mask_gan, x1, x2, y)
                if mean_bottom < 0.6:
                    for i in range(x1, x2):
                        field_mask_gan[int(y(i)):image_shape[0], i] = 0
    return field_mask_gan


def post_process_mask(img, field_mask, GAN_mask_im, GAN_line_im):
    GAN_mask_im = cv2.resize(GAN_mask_im, (field_mask.shape[1], field_mask.shape[0]))
    GAN_line_im = cv2.resize(GAN_line_im, (field_mask.shape[1], field_mask.shape[0]))
    gray_mask = cv2.cvtColor(GAN_mask_im, cv2.COLOR_BGR2GRAY)
    gray_line = cv2.cvtColor(GAN_line_im, cv2.COLOR_BGR2GRAY)
    image_shape = gray_mask.shape
    lower = np.array([0])
    upper = np.array([100])
    field_mask_gan = 255 - cv2.inRange(gray_mask, lower, upper)
    field_line_gan = 255 - cv2.inRange(gray_line, lower, upper)
    color_bottom_line(field_line_gan, field_mask_gan)

    field_mask_final = field_mask_gan
    field_mask_final[:int(2 * image_shape[0] / 3), :] = field_mask[:int(2 * image_shape[0] / 3), :]

    field_in = np.zeros_like(img)
    field_in[field_mask_final == 255, :] = img[field_mask_final == 255, :]

    return field_in, field_mask_final


def save_fields(image_files, mask_gan, field_path, frame_path, gan_path, n_im):
    if mask_gan:
        # run pix2pix model
        subprocess.run(
            ['python3.7', '../other_utils/pytorch-two-GAN/test_two_pix2pix.py', '--dataroot', frame_path,
             '--which_direction', 'AtoB', '--model', 'test', '--name', 'soccer_seg_detection_pix2pix', '--output_nc',
             '1', '--dataset_mode', 'single', '--which_model_netG', 'unet_256', '--loadSize', '256',
             '--checkpoints_dir', '../../checkpoints_runs/checkpoint_pix2pix',
             '--which_epoch', 'latest', '--results_dir', gan_path,
             '--aspect_ratio', '1.77', '--how_many', str(n_im)])

        GAN_mask_images = []
        GAN_line_images = []
        dir = gan_path + '/soccer_seg_detection_pix2pix/test_latest/images/'
        for path in natsorted(os.listdir(dir), alg=ns.PATH | ns.IGNORECASE):
            if 'fake_B' in path:
                path = os.path.join(dir, path)
                im = cv2.imread(path)
                GAN_mask_images.append(im)
            if 'fake_D' in path:
                path = os.path.join(dir, path)
                im = cv2.imread(path)
                GAN_line_images.append(im)

    for f in range(n_im):
        im = cv2.imread(image_files[f])
        field_in, field_mask = field_selection(im)  # select field with contour detection
        if mask_gan:
            GAN_mask_im = GAN_mask_images[f]
            GAN_line_im = GAN_line_images[f]
            field_in, field_mask = post_process_mask(im, field_mask, GAN_mask_im, GAN_line_im)  # combine pix2pix
            # detection with previous detection

        cv2.imwrite(os.path.join(field_path, str(f) + '_field_in.png'), field_in)
        cv2.imwrite(os.path.join(field_path, str(f) + '_field_mask.png'), field_mask)


def player_detection(image, score_thres, model, max_w, max_h, device):
    player_detections = []
    detection_scores = []
    with torch.no_grad():
        x = [T.ToTensor()(image).to(device)]
        output = model(x)[0][0]
        scores = output['scores']
        labels = output['labels']
        boxes = output['boxes']
        for i in range(len(scores)):
            if labels[i]==1 and scores[i]>score_thres :
                xmin,ymin,xmax,ymax = int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])
                if 0 < xmax-xmin < max_w and 0 < ymax-ymin < max_h:
                    player_detections.append([xmin,ymin,xmax,ymax])
                    detection_scores.append(scores[i])
    return player_detections, detection_scores


def save_player_dets(path, image_files, score_thres, model_detection, max_w, max_h, device, n_im):
    for f in range(n_im):
        im = Image.open(image_files[f]).convert('RGB')
        player_detections, detection_scores = player_detection(im, score_thres, model_detection, max_w, max_h, device)
        annotation_path = os.path.join(path, str(f) + '.txt')
        with open(annotation_path, 'w') as det_file :
            for b,bbox in enumerate(player_detections) :
                det_file.write(str(float(detection_scores[b].to('cpu'))) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) +
                               ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')


def load_player_dets(path, n_im, f, use_field_detection, field_path, field_in_thres, get_score=True):

    player_detections = []
    detection_scores = []
    field_mask = cv2.imread(os.path.join(field_path, str(f) + '_field_mask.png'))
    annotation_path = os.path.join(path, str(f) + '.txt')
    with open(annotation_path, 'r') as det_file :
        lines = det_file.readlines()
        for line in lines :
            split = line.split()
            if get_score :
                detection_score, xmin, ymin, xmax ,ymax = float(split[0]), float(split[1]), float(split[2]), float(split[3]), float(split[4])
                if use_field_detection:
                    if np.mean(field_mask[int(ymin):int(ymax), int(xmin):int(xmax)])/255. > field_in_thres :
                        detection_scores.append(detection_score)
                        player_detections.append([xmin,ymin,xmax,ymax])
                else :
                    detection_scores.append(detection_score)
                    player_detections.append([xmin,ymin,xmax,ymax])
            else :
                xmin, ymin, xmax ,ymax = float(split[0]), float(split[1]), float(split[2]), float(split[3])
                if use_field_detection :
                    if np.mean(field_mask[int(ymin):int(ymax), int(xmin):int(xmax)])/255. > field_in_thres :
                        player_detections.append([xmin,ymin,xmax,ymax])
                else :
                    player_detections.append([xmin,ymin,xmax,ymax])
    if get_score :
        return player_detections, detection_scores
    else :
        return player_detections


def collect_avg_bbox_dims(player_detections,im_shape, rem_outliers_area, n_parts = 3) :

    areas = []
    parts = []

    y_list = [i*im_shape[0]/n_parts for i in range(n_parts+1)]

    prop_list = []
    area_list = [[] for i in range(n_parts)]

    for i in range(len(player_detections)):
        n = len(player_detections[i])
        areas.append(np.zeros(n))
        parts.append(np.zeros(n))
        for j in range(n) :
            det = player_detections[i][j]
            area = (det[2]-det[0])*(det[3]-det[1])
            areas[i][j] = area
            prop = (det[3]-det[1])/(det[2]-det[0])
            prop_list.append(prop)
            y = (det[3]+det[1])/2
            f = True
            part = 0
            while f and part < len(y_list):
                if y_list[part] <= y < y_list[part+1]:
                    f = False
                else :
                    part += 1
            parts[i][j] = int(part)
            area_list[part].append(area)

    for i in range(n_parts):
        if len(area_list[i]) == 0:
            if i == 0 :
                if len(area_list[1]) == 0:
                    area_list[i] = area_list[2]
                else :
                    area_list[i] = area_list[1]
            else :
                if not len(area_list[i-1]) == 0:
                    area_list[i] = area_list[i-1]
                else :
                    area_list[i] = area_list[i+1]

    area_list = np.array(area_list)

    if rem_outliers_area :

        medians = [np.median(area_l) for area_l in area_list]
        for i in range(len(player_detections)) :
            n = len(player_detections[i])
            mask = np.ones(n, dtype = bool)
            for j in range(n):
                p = int(parts[i][j])
                dev = np.abs(areas[i][j]-medians[p])/medians[p]
                if dev > 1 :
                    mask[j] = False
            player_detections[i] = list(np.array(player_detections[i])[mask])

    area_avg = [np.mean(area_list[i]) for i in range(n_parts)]
    prop_avg = np.mean(prop_list)

    return player_detections, area_avg, prop_avg


def player_segmentation(img, avg_bbox_dims, prop_avg):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([15,50,50])
    upper_green = np.array([70,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours_ext,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
    max_w = 150
    min_w = 5
    max_h = 150
    min_h = 5
    detections = []
    n_parts = len(avg_bbox_dims)
    y_list = [i*img.shape[0]/n_parts for i in range(n_parts+1)]
    for c,contour in enumerate(contours):
        #cv2.drawContours(img, contours, c, (0,255,0), 2)
        x,y,w,h = cv2.boundingRect(contour)
        f = True
        i = 0
        yc = y + h/2
        while f and i < len(y_list) :
            if y_list[i] <= yc < y_list[i+1] :
                f = False
            else :
                i += 1
        area = w*h
        target_area = avg_bbox_dims[i]
        if min_w < w < max_w and max(min_h,0.5*w) < h < min(max_h,5*w) and target_area/4 < area < 1.5*target_area :
            xc = x + w/2
            yc = y + h/2.
            new_h = np.sqrt(target_area*prop_avg)
            new_w = np.sqrt(target_area/prop_avg)
            xmin,xmax,ymin,ymax = max(0,xc-new_w/2), min(img.shape[1],xc+new_w/2), max(0,yc-new_h/2), min(img.shape[0],yc+new_h/2)
            det = [xmin,ymin,xmax,ymax]
            detections.append(det)
    return detections


def initialize_pose_estimator():
    initialize_parameters()
    pose_model_path = '../other_utils/lighttrack/weights/CPN101/CPN_snapshot_293.ckpt'
    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights(pose_model_path)
    return pose_estimator


def correct_bbox_dims(detections, avg_bbox_dims, prop_avg, im_shape):
    '''
    Correct the bbox dimensions using the the bbox detcted in the same parts of the field.
    '''
    corrected_detections = []
    n_parts = len(avg_bbox_dims)
    y_list = [i*im_shape[0]/n_parts for i in range(n_parts+1)]
    for det in detections:
        x = (det[2]+det[0])/2
        y = (det[3]+det[1])/2
        h = det[3]-det[1]
        w = det[2]-det[0]
        f = True
        i = 0
        while f and i < len(y_list) :
            if y_list[i] <= y < y_list[i+1] :
                f = False
            else :
                i += 1
        area = w*h
        target_area = avg_bbox_dims[i]
        new_h = np.sqrt(target_area*prop_avg)
        new_w = np.sqrt(target_area/prop_avg)
        xmin,xmax,ymin,ymax = max(0,x-new_w/2), min(im_shape[1],x+new_w/2), max(0,y-new_h/2), min(im_shape[0],y+new_h/2)
        det = [xmin,ymin,xmax,ymax]
        corrected_detections.append(det)

    return corrected_detections


def segmentation_correction(n_im, images, path_segs, player_detections_list, score_detection_list,
        avg_bbox_dims, prop_avg, use_SR, use_pose_estimator, pose_estimator, thres_pose_estimator,
        field_path, image_files, visualization, show_images):

    output_detections = []
    output_scores = []
    seg_bbox_list_list = []

    for f in range(n_im):
        field_in = cv2.imread(os.path.join(field_path, str(f) + '_field_in.png'))
        seg_bbox_list = player_segmentation(field_in, avg_bbox_dims, prop_avg)
        seg_bbox_list = torch.tensor(seg_bbox_list).type(torch.FloatTensor)
        scores = torch.tensor([1.]*len(seg_bbox_list))
        if len(seg_bbox_list) > 0:
            keep = torchvision.ops.boxes.nms(seg_bbox_list, scores, 0.1)
            seg_bbox_list = seg_bbox_list[keep].type(torch.IntTensor).numpy()
        seg_bbox_list_list.append(seg_bbox_list)

        annotation_path = os.path.join(path_segs, str(f) + '.txt')
        with open(annotation_path, 'w') as det_file:
            for b,bbox in enumerate(seg_bbox_list):
                det_file.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')

    for f in range(n_im):

        player_detections = player_detections_list[f]
        player_scores = score_detection_list[f]
        im = images[f]
        seg_bbox_list = seg_bbox_list_list[f]

        interAreas = np.array([[iou(player_det,seg_bbox) for player_det in player_detections] for seg_bbox in seg_bbox_list])
        intersection_list = interAreas > 0.1
        corrected_detections = [seg_bbox_list[i] for i in range(len(intersection_list)) if not True in intersection_list[i]]

        if use_pose_estimator: # Check the score given by a pose estimator on the detected blobs
            corrected_detections_pose = []
            if not use_SR:
                for d,det in enumerate(corrected_detections) :
                    player = [det[0],det[1],det[2]-det[0],det[3]-det[1]]
                    data = {"img_id": 0,
                            "det_id":  0,
                            "track_id": None,
                            "imgpath": image_files[f],
                            "bbox": player,
                            "matching": None}
                    inf, visual_feat = inference_feat_keypoints(pose_estimator, data)
                    score = inf[0]["score"]
                    if score > thres_pose_estimator :
                        keypoints = inf[0]["keypoints"]
                        bbox = get_bbox_from_keypoints(keypoints, im.shape)
                        bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
                        corrected_detections_pose.append(bbox)

            # if use_SR: # Use Super-Resolution  on the detected blobs before applying the pose estimator

            #     for d,det in enumerate(corrected_detections) :
            #         path_SR_inputs = 'ESRGAN/LR/'
            #         path_SR_outputs = 'ESRGAN/results/'
            #         if not os.path.exists(path_SR_inputs):
            #             os.mkdir(path_SR_inputs)
            #         if os.path.exists(path_SR_outputs):
            #             os.mkdir(path_SR_outputs)
            #         #cut = enlargeBbox(det, 1., im.shape)
            #         im_cut = field_in[cut[1]:cut[3],cut[0]:cut[2],:]
            #         cv2.imwrite(path_SR_inputs+'in.png',im_cut)
            #         test_SR()
            #         SR_cut = cv2.imread(path_SR_outputs+'out.png')
            #
            #         player = [0,0,SR_cut.shape[1],SR_cut.shape[0]]
            #         data = {"img_id": 0,
            #                 "det_id":  0,
            #                 "track_id": None,
            #                 "imgpath": path_SR_outputs+str(d)+'.png',
            #                 "bbox": player,
            #                 "matching": None}
            #
            #         inf, visual_feat = inference_feat_keypoints(pose_estimator, data)
            #         score = inf[0]["score"]
            #         if score > 0.5 :
            #             keypoints = inf[0]["keypoints"]
            #             bbox = get_bbox_from_keypoints(keypoints, im.shape)
            #             bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
            #             det_cut = np.array(bbox)/4.
            #             bbox = [cut[0] + det_cut[0], cut[1] + det_cut[1], cut[0] + det_cut[2], cut[1] + det_cut[3]]
            #             corrected_detections_pose.append(bbox)

            corrected_detections_pose = correct_bbox_dims(corrected_detections_pose, avg_bbox_dims, prop_avg, im.shape)
            corrected_detections = corrected_detections_pose
            corrected_score = [2]*len(corrected_detections)
            output_detections.append(corrected_detections + player_detections)
            output_scores.append(corrected_score + player_scores)

            if visualization:
                for det in corrected_detections:
                    cv2.rectangle(show_images[f], (int(det[0]), int(det[1]),int(det[2]-det[0]), int(det[3]-det[1])), (0, 255, 0), 3)

    return output_detections, output_scores

