
import os
import numpy as np
import json
from pprint import pprint
import argparse
import sys
import cv2
import random
sys.path.append('/home/u176443/Documents/ARPE/lighttrack')
from detector.detector_yolov3 import *
import torchvision
import torch
from pascal_voc_writer import Writer
import shutil

def player_detection(image, score_thres, model, max_w, max_h):
    player_detections = []
    detection_scores = []
    with torch.no_grad():
        image = np.transpose(image/255.,(2,0,1))
        x = torch.tensor([image]).type(torch.FloatTensor).to(device)
        output = model(x)[0]
        scores = output['scores']
        labels =  output['labels']
        boxes = output['boxes']
        for i in range(len(scores)):
            if labels[i]==1 and scores[i]>score_thres :
                xmin,ymin,xmax,ymax = int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])
                if xmax-xmin < max_w and ymax-ymin < max_h :
                    xmin,ymin,xmax,ymax = enlarge_bbox([xmin,ymin,xmax,ymax],0.1, image.shape)
                    player_detections.append([xmin,ymin,xmax-xmin,ymax-ymin])
                    detection_scores.append(scores[i])
    return(player_detections, detection_scores)

### 2e DATASET

video_folder = '../data/TV_soccer/selection'
annotation_folder = '../data/TV_soccer/val/annotation'
frame_folder = '../data/TV_soccer/val/frame'
max_w = 60
max_h = 150
score_thres = 0.8

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# Load all games

for videofile in os.listdir(video_folder):

    print(videofile)

    t1 = time.time()

    cap = cv2.VideoCapture(os.path.join(video_folder,videofile))
    game_id = os.path.splitext(os.path.split(videofile)[-1])[0]
    p = os.path.join(frame_folder,game_id)
    if not os.path.isdir(p) :
        os.mkdir(p)
        p = os.path.join(annotation_folder,game_id)
        os.mkdir(p)

        # Create MultiTracker object
        multiTracker = cv2.MultiTracker_create()

        # Initialize MultiTracker
        trackerType = "CSRT"
        success = True
        frame_id = 0
        image = None
        multiTracker = None
        intersection_thres = 0.2

        key_frame = False
        total_hard_neg = 0
        total_hard_pos = 0

        while success :

            if (frame_id - 1) % 10 == 0 :
                key_frame = True
                multiTracker = cv2.MultiTracker_create()
                for det in player_detections :
                    tracker = cv2.TrackerCSRT_create()
                    multiTracker.add(tracker, image, tuple(det))

            success,image = cap.read()

            if success :

                field_in,field_mask = field_selection(image)

                tracker_bbox_list = []
                image_shape = image.shape
                if multiTracker is not None :
                    tracker_success, boxes = multiTracker.update(image)
                    if tracker_success:
                        for box in boxes :
                            bbox = [int(v) for v in box]
                            (x, y, w, h) = bbox
                            tracker_bbox_list.append(bbox)
                            #cv2.rectangle(image, (x, y), (x + w, y + h),(0, 255, 0), 2)

                frame_path = os.path.join(frame_folder,game_id, str(frame_id) + '.jpg')
                annotation_path = os.path.join(annotation_folder,game_id, str(frame_id) + '.xml')
                detection_bbox_list = []
                player_detections, detection_scores = player_detection(image,score_thres, model, max_w, max_h)



                if len(player_detections) > 0 :

                    hard_pos = []

                    cv2.imwrite(frame_path, image)

                    writer = Writer(frame_path, image_shape[1], image_shape[0])

                    for b,bbox in enumerate(player_detections) :
                        (x, y, w, h) = bbox
                        if np.mean(field_mask[y:y+h, x:x+w])/255. > 0.2 :
                            detection_bbox_list.append(bbox)
                            cv2.rectangle(image, (x, y), (x + w, y + h),(255, 0, 0), 2)
                            writer.addObject('player', bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], difficult = float(detection_scores[b]))
                        else :
                            hard_pos.append(bbox)

                    interAreas = np.array([[iou(detection_bbox,tracker_bbox) for detection_bbox in detection_bbox_list] for tracker_bbox in tracker_bbox_list])
                    check_intersection_list = interAreas < intersection_thres
                    hard_negs = [tracker_bbox_list[i] for i in range(len(check_intersection_list)) if False not in check_intersection_list[i]]

                    for bbox in hard_negs :
                        (x, y, w, h) = bbox
                        writer.addObject('player', bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], difficult = 1)
                        cv2.rectangle(image, (x, y), (x + w, y + h),(0, 0, 255), 2)

                    writer.save(annotation_path)

                total_hard_neg+= len(hard_negs)
                total_hard_pos+= len(hard_pos)

                frame_id+=1
                key_frame = False
                #cv2.imshow('frame',image)
                #cv2.waitKey(100)
        print(total_hard_neg)
        print(total_hard_pos)

        t2 = time.time()
        print(t2-t1)

        cap.release()
        cv2.destroyAllWindows()
