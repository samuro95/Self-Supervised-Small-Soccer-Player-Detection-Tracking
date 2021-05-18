import os
import sys
import collections
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
import torch
import torchvision
import cv2
import numpy as np
import random
import pathlib

def resize_bbox(bbox,scale_transform):
    min_x, min_y, max_x, max_y = bbox
    wc = (max_x + min_x)/2
    hc = (max_y + min_y)/2
    w = max_x - min_x
    h = max_y - min_y
    new_w = w*scale_transform
    new_h = h*scale_transform
    min_x = wc - new_w/2.
    max_x = wc + new_w/2.
    min_y = hc - new_h/2.
    max_y = hc + new_h/2.
    new_bbox = [min_x, min_y, max_x, max_y]
    return(new_bbox)

class SoccerDataset():
    """
    Args:
        root (string, optional): root directory of the VOC Dataset.
        split_f (string, optional): list of the frame fo the training dataset
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    CLASSES = (
        "__background__ ",
        "player")

    def __init__(self,
                 train_image_files='../../data/SoccerNet/train_frame_list.txt',
                 train_annotation_files='../../data/SoccerNet/train_annotation_r1_list.txt',
                 test_dataset_name='TV_soccer',
                 test_image_files='../../data/data/TV_soccer/test_frame_list.txt',
                 test_annotation_files='../../data/data/TV_soccer/test_annotation_list.txt',
                 transform=None,
                 train=False,
                 weight_loss=False,
                 weight_track = 1,
                 weight_seg = 1,
                 scale_transform = 1,
                 use_field_detection = False,
                 only_det = False,
                 train_list_stride = 10,
                 round_2 = False,
                 train_annotation_files_r2='../../data/SoccerNet/train_annotation_r2_list.txt',
                 visualize = False,
                 min = 0,
                 max = -1):

        self.test_dataset_name = test_dataset_name
        self.train = train
        self.visualize = visualize
        self.transform = transform
        self.scale_transform = scale_transform
        self.weight_seg = weight_seg
        self.weight_track = weight_track
        self.round_2 = round_2
        self.train = train
        self.use_field_detection = use_field_detection

        if train:

            with open(train_image_files, "r") as f:
                self.images = [x.strip() for x in f.readlines()][min:max:train_list_stride]

            if not round_2 :
                with open(train_annotation_files, "r") as f:
                    self.annotations = [x.strip() for x in f.readlines()][min:max:train_list_stride]
            else :
                with open(train_annotation_files_r2, "r") as f:
                    self.annotations = [x.strip() for x in f.readlines()][min:max:train_list_stride]

        else:
            with open(os.path.join(test_image_files), "r") as f:
                self.images = [x.strip() for x in f.readlines()][min:max]
            with open(os.path.join(test_annotation_files), "r") as f:
                self.annotations = [x.strip() for x in f.readlines()][min:max]

        img = Image.open(self.images[0]).convert('RGB')
        self.image_shape = self.transform(img.convert('RGB')).shape

        parts = list(pathlib.Path().absolute().parts)[:-2]
        parts.append('data')
        data_path = pathlib.Path(*parts)
        intermediate_path = os.path.join(data_path,'intermediate', test_dataset_name)
        if not os.path.exists(intermediate_path):
            os.mkdir(intermediate_path)

    def rescale_img(self, img):
        w = self.image_shape[2]
        h = self.image_shape[1]
        desired_h = h*self.current_scale_transform
        desired_w = w*self.current_scale_transform
        img = torchvision.transforms.Resize([int(desired_h), int(desired_w)])(img)
        w_pad = (w - w*self.current_scale_transform)/2.
        h_pad = (h - h*self.current_scale_transform)/2.
        img = torchvision.transforms.Pad((int(w_pad),int(h_pad)))(img)
        return(img)

    def rescale_bbox(self, bbox):
        w = self.image_shape[2]
        h = self.image_shape[1]
        bbox = np.array(bbox)*self.current_scale_transform
        target_w = w*self.current_scale_transform
        target_h = h*self.current_scale_transform
        w_pad = (w - target_w)/2.
        h_pad = (h - target_h)/2.
        bbox = bbox + np.array([w_pad,h_pad,w_pad,h_pad])
        return(bbox)

    def __getitem__(self, index):

        #try :

        annotation_path = self.annotations[index]
        img_path = self.images[index]
        target = self.parse_voc_xml(ET.parse(annotation_path).getroot())
        annotation = target['annotation']

        try:
            objects = annotation['object']
        except:
            objects = []
        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        classes = []
        area = []
        weight = []
        iscrowd = []

        img = Image.open(img_path).convert('RGB')

        if self.train:
            self.current_scale_transform = random.uniform(self.scale_transform,1.)
        else:
            self.current_scale_transform = self.scale_transform
        img = self.rescale_img(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.visualize:
            imshow = np.array(img).transpose(1, 2, 0)
            imshow = cv2.cvtColor(imshow, cv2.COLOR_RGB2BGR)

        for obj in objects:
            bbox = obj['bndbox']
            if not False in [el in bbox.keys() for el in ['xmin', 'ymin', 'xmax', 'ymax']] :
                try:
                    bbox = [int(float(bbox[n])) for n in ['xmin', 'ymin', 'xmax', 'ymax']]
                except:
                    bbox = [int(float(bbox[n][7:-1])) for n in ['xmin', 'ymin', 'xmax', 'ymax']]
                bbox = self.rescale_bbox(bbox)
                method = obj['name']
                if method == 'track' and self.weight_track > 0 and self.train:
                    weight.append(float(self.weight_track))
                    boxes.append(bbox)
                    classes.append(1)
                    area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    iscrowd.append(False)
                    if self.visualize:
                        det = bbox
                        cv2.rectangle(imshow, (int(det[0]), int(det[1]),int(det[2]-det[0]), int(det[3]-det[1])), (255, 0, 0), 2)
                elif method == 'seg' and self.weight_seg > 0 and self.train:
                    weight.append(float(self.weight_seg))
                    boxes.append(bbox)
                    classes.append(1)
                    area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    iscrowd.append(False)
                    if self.visualize:
                        det = bbox
                        cv2.rectangle(imshow, (int(det[0]), int(det[1]),int(det[2]-det[0]), int(det[3]-det[1])), (0, 255, 0), 2)
                elif self.train:
                    w = float(obj["difficult"])
                    if method != 'non_player' and ((self.round_2 and w > 0.7) or not self.round_2) :
                        weight.append(w)
                        boxes.append(bbox)
                        classes.append(1)
                        area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                        iscrowd.append(False)
                        if self.visualize:
                            det = bbox
                            cv2.rectangle(imshow, (int(det[0]), int(det[1]),int(det[2]-det[0]), int(det[3]-det[1])), (255, 255, 255), 2)
                else:
                    weight.append(1.)
                    boxes.append(bbox)
                    classes.append(1)
                    area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    iscrowd.append(False)
                    if self.visualize:
                        det = bbox
                        cv2.rectangle(imshow,
                                      (int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])),
                                      (255, 255, 255), 2)
        if self.visualize:
            print(index)
            cv2.imwrite('output/'+str(index)+'.png',imshow*255)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        weight = torch.as_tensor(weight)
        iscrowd = torch.as_tensor(iscrowd)
        image_id = torch.as_tensor([int(index)])

        output = {}
        output["boxes"] = boxes
        output["labels"] = classes
        output["image_id"] = image_id
        output["area"] = area
        output["weight"] = weight
        output["iscrowd"] = iscrowd

        return img, output

        # except Exception as e:
        #
        #     print(e)
        #
        #     index = index - 1 if index > 0 else index + 1
        #     return self.__getitem__(index)

    def __len__(self):
        return len(self.annotations)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
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


class Annotater_Dataset():

    def __init__(self,
                 image_files='/../../data/SoccerNet/train_frame_list.txt',
                 transform=None,
                 image_shape = (720,1280),
                 data_name = 'SoccerNet',
                 use_field_detection = True,
                 field_files='../../data/SoccerNet/train_frame_list.txt',):

        self.data_name = data_name

        with open(image_files, "r") as f:
            self.images = [x.strip() for x in f.readlines()]

        self.transform = transform

        img = Image.open(self.images[0]).convert('RGB')
        self.image_shape = self.transform(img.convert('RGB')).shape

        self.use_field_detection = use_field_detection

        with open(field_files, "r") as f:
            self.field_paths = [x.strip() for x in f.readlines()]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """

        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.use_field_detection :
            field_path = self.field_paths[index]
            field_mask = cv2.imread(field_path)
        else :
            field_mask = None

        return(img,field_mask,img_path)

    def __len__(self):
        return len(self.images)
