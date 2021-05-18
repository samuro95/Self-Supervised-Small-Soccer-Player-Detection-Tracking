from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import misc as misc_nn_ops
#from torchvision.ops import MultiScaleRoIAlign
from poolers import MultiScaleRoIAlign
from torchvision.models.utils import load_state_dict_from_url
from generalized_rcnn import GeneralizedRCNN
from rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from roi_head import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from backbone_utils import resnet_fpn_backbone, detnet_fpn_backbone
from track_embed_head import TrackHeads
from torchsummary import summary

""" Modified from 
TORCHVISION.MODELS.DETECTION
"""

class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.
    """

    def __init__(self,
                 backbone, n_channel_backbone = 5,
                 num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 #min_size=720, max_size=1280,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.5,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 anchor_sizes = [32,64,128,256,512],
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.4, box_detections_per_img=30,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 weight_loss=False,
                 use_soft_nms=False,
                 use_context=False,
                 use_track_branch=False):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            ratios = ((0.5,1.0, 2.0),)
            aspect_ratios = ratios * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh, weight_loss)

        if box_roi_pool is None:

            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0,1,2,3],
                output_size=7,
                sampling_ratio=2)

            if n_channel_backbone == 6 :

                box_roi_pool = MultiScaleRoIAlign(
                    featmap_names=[0,1,2,3,4],
                    output_size=7,
                    sampling_ratio=2)

        representation_size1 = 1024
        representation_size2 = 1024
        track_embedding_size = 1024

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            if use_context :
                box_head = TwoMLPHead(
                    2*out_channels * resolution ** 2,
                    representation_size1,representation_size2)
            else :
                box_head = TwoMLPHead(
                    out_channels * resolution ** 2,
                    representation_size1,representation_size2)

        if use_track_branch :
            if use_context :
                track_embedding = TwoMLPHead(
                    2*out_channels * resolution ** 2,
                    representation_size1, track_embedding_size)
            else :
                track_embedding = TwoMLPHead(
                    out_channels * resolution ** 2,
                    representation_size1, track_embedding_size)
        else :
            track_embedding = None

        if box_predictor is None:
            box_predictor = FastRCNNPredictor(
                representation_size1,
                num_classes)

        if num_classes > 2 :
            use_soft_nms = False

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights, box_score_thresh, box_nms_thresh, box_detections_per_img,
            weight_loss=weight_loss, use_soft_nms = use_soft_nms,  use_context=use_context)

        if use_track_branch :
            track_heads = TrackHeads(
                box_roi_pool,
                box_head,
                box_predictor,
                box_fg_iou_thresh, box_bg_iou_thresh,
                box_batch_size_per_image, box_positive_fraction,
                bbox_reg_weights,
                weight_loss=False,
                use_context=False,
                track_embedding=track_embedding)
        else :
            track_heads = None

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, track_heads, transform, n_channel_backbone)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size1,representation_size2):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size1)
        self.fc7 = nn.Linear(representation_size1, representation_size2)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class TrackingEmbedding(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
}


def fasterrcnn_resnet50_fpn(pretrained=False,
                            num_classes=91, pretrained_backbone=True, weight_loss=False,
                            detection_score_thres = 0.05, use_soft_nms = False, nms_thres = 0.4,
                            anchor_sizes = [32,64,128,256,512], n_channel_backbone = 5,
                            use_context=False, use_track_branch = False, **kwargs):

    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, n_channel_backbone, first_layer_out = 'layer1')
    if pretrained:
        model = FasterRCNN(backbone, num_classes = 91, weight_loss=weight_loss,
        box_score_thresh = detection_score_thres, use_soft_nms = use_soft_nms,
        box_nms_thresh = nms_thres, **kwargs)
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=True)
        model.load_state_dict(state_dict)
    else :
        model = FasterRCNN(backbone, num_classes = num_classes, box_score_thresh = detection_score_thres,
            anchor_sizes = anchor_sizes, weight_loss=weight_loss, use_soft_nms = use_soft_nms,
            use_context = use_context, use_track_branch = use_track_branch, **kwargs)
    return model

def fasterrcnn_detnet59_fpn(num_classes=91, pretrained_backbone=True, weight_loss=False,
                            detection_score_thres = 0.05, use_soft_nms = False, nms_thres = 0.4,
                            anchor_sizes = [32,64,128,256,512], use_context=False, use_track_branch = False, **kwargs):
    backbone = detnet_fpn_backbone('detnet59', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes = num_classes, box_score_thresh = detection_score_thres,
            anchor_sizes = anchor_sizes, weight_loss=weight_loss, use_soft_nms = use_soft_nms,
            n_channel_backbone = 6, use_context = use_context, box_nms_thresh = nms_thres, use_track_branch = use_track_branch,**kwargs)
    return model

def fasterrcnn_resnet18_fpn(num_classes=91, pretrained_backbone=True, weight_loss=False,
                            detection_score_thres = 0.05, use_soft_nms = False,nms_thres = 0.4,
                            anchor_sizes = [32,64,128,256,512], n_channel_backbone = 5,
                            use_context=False, use_track_branch = False, **kwargs):

    backbone = resnet_fpn_backbone('resnet18', pretrained_backbone, n_channel_backbone, first_layer_out = 'layer1')

    model = FasterRCNN(backbone, num_classes = num_classes, use_soft_nms = use_soft_nms,
    n_channel_backbone = n_channel_backbone, weight_loss=weight_loss,
    box_score_thresh = detection_score_thres, use_context = use_context, use_track_branch = use_track_branch, **kwargs)
    return model

def fasterrcnn_resnet8_fpn(num_classes=91, pretrained_backbone=True, weight_loss=False,
                            detection_score_thres = 0.05, use_soft_nms = False,nms_thres = 0.4,
                            anchor_sizes = [32,64,128,256,512], n_channel_backbone = 5,
                            use_context=False, use_track_branch = False, **kwargs):

    backbone = resnet_fpn_backbone('resnet8', pretrained_backbone, n_channel_backbone, first_layer_out = 'layer1')

    model = FasterRCNN(backbone, num_classes = num_classes, use_soft_nms = use_soft_nms,
    n_channel_backbone = n_channel_backbone, weight_loss=weight_loss,
    box_score_thresh = detection_score_thres, use_context = use_context, use_track_branch = use_track_branch, **kwargs)
    return model

def fasterrcnn_resnet34_fpn(num_classes=91, pretrained_backbone=True, weight_loss=False,
                            detection_score_thres = 0.05, use_soft_nms = False,nms_thres = 0.4,
                            anchor_sizes = [32,64,128,256,512], n_channel_backbone = 5,
                            use_context=False, use_track_branch = False, **kwargs):
    backbone = resnet_fpn_backbone('resnet34', pretrained_backbone, n_channel_backbone)
    model = FasterRCNN(backbone, num_classes = num_classes, use_soft_nms = use_soft_nms,
    n_channel_backbone = n_channel_backbone, weight_loss=weight_loss,
    box_score_thresh = detection_score_thres, use_context = use_context, box_nms_thres = nms_thres, use_track_branch = use_track_branch, **kwargs)
    return model

def fasterrcnn_resnet101_fpn(num_classes=91, pretrained_backbone=True, weight_loss=False,
                            detection_score_thres = 0.05, use_soft_nms = False,
                            anchor_sizes = [32,64,128,256,512], n_channel_backbone = 5,
                            use_context=False, use_track_branch = False, **kwargs):
    backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, n_channel_backbone)
    model = FasterRCNN(backbone, num_classes = num_classes, use_soft_nms = use_soft_nms,
    n_channel_backbone = n_channel_backbone, weight_loss=weight_loss,
    box_score_thresh = detection_score_thres, use_context = use_context, box_nms_thres = nms_thres, use_track_branch = use_track_branch, **kwargs)
    return model
