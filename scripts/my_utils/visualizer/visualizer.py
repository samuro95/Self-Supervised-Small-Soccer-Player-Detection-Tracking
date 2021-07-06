'''
 visualizer.py
 Visualizer for Detection, Human Pose Estimation, Segmentation, etc
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    Created on June 19th, 2018
'''
import sys, os
sys.path.insert(0, os.path.abspath("../detect_to_standard/"))
from detection_visualizer import *

sys.path.append(os.path.abspath("../keypoint_to_standard/"))
from keypoint_visualizer import *
import torchvision
from PIL import Image

draw_threshold = 0.4

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

def show_all_from_standard_json(json_file_path, classes, joint_pairs, joint_names, rescale_img_factor = 1., img_folder_path = None, output_folder_path = None, display_pose = False, flag_track= False):
    # Visualizing: Detection + Pose Estimation
    dets = read_json_from_file(json_file_path)

    for det in dets:
        python_data = det

        if img_folder_path is None:
            img_path = os.path.join(python_data["image"]["folder"], python_data["image"]["name"])
        else:
            img_path = os.path.join(img_folder_path, python_data["image"]["name"])
        print(img_path)
        if is_image(img_path):
            img = Image.open(image_path).convert('RGB')
            img = rescale_img(im,rescale_img_factor)
            img = np.array(img).transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        candidates = python_data["candidates"]
        for candidate in candidates:
            bbox = np.array(candidate["det_bbox"]).astype(int)
            if flag_track is True:
                track_id = candidate["track_id"]
                img = draw_bbox(img, bbox, classes, track_id = track_id)
            else:
                #img = draw_bbox(img, bbox, score, classes)
                img = draw_bbox(img, bbox, classes, -1, python_data["image"]["id"][0])  #for lighttrack

            if display_pose :
                pose_keypoints_2d = candidate["pose_keypoints_2d"]
                joints = reshape_keypoints_into_joints(pose_keypoints_2d)

                if flag_track is True:
                    track_id = candidate["track_id"]
                    img = show_poses_from_python_data(img, joints, joint_pairs, joint_names, track_id = track_id)
                    #img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)
                else:
                    img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)

        if output_folder_path is not None:
            create_folder(output_folder_path)
            img_output_path = os.path.join(output_folder_path, python_data["image"]["name"])
            cv2.imwrite(img_output_path, img)
    return


def show_all_from_dict(keypoints_list_list, bbox_dets_list_list, classes, joint_pairs, joint_names, rescale_img_factor = 1., img_folder_path = None, output_folder_path = None, display_pose = False, flag_track= False, flag_method = False):

    for i,bbox_dets_list in enumerate(bbox_dets_list_list) :

        bbox_dets_init = bbox_dets_list[0]
        img_path = bbox_dets_init["imgpath"]
        img_name =  os.path.basename(img_path)

        if is_image(img_path):
            img = Image.open(img_path).convert('RGB')
            img = rescale_img(img,rescale_img_factor)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for j,candidate in enumerate(bbox_dets_list) :

            bbox = np.array(candidate["bbox"]).astype(int)
            method = candidate["method"]

            if flag_track is True:
                track_id = candidate["track_id"]
                img = draw_bbox(img, bbox, classes, method, track_id = track_id, flag_method = flag_method)
            else:
                #img = draw_bbox(img, bbox, score, classes)
                img = draw_bbox(img, bbox, classes, method, -1, candidate["img_id"])  #for lighttrack

            if display_pose :
                pose_dict = keypoints_list_list[i][j]
                pose_keypoints_2d = pose_dict["keypoints"]
                joints = reshape_keypoints_into_joints(pose_keypoints_2d)

                if flag_track is True:
                    track_id = candidate["track_id"]
                    img = show_poses_from_python_data(img, joints, joint_pairs, joint_names, track_id = track_id)
                    #img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)
                else:
                    img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)

        if output_folder_path is not None:
            create_folder(output_folder_path)
            img_output_path = os.path.join(output_folder_path,img_name)
            cv2.imwrite(img_output_path, img)
    return


def make_video_from_images(img_paths, outvid_path, fps=25, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for ct, img_path in enumerate(img_paths):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        img = imread(img_path)
        if img is None:
            print(img_path)
            continue
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid_path, fourcc, float(fps), size, is_color)

        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    if vid is not None:
        vid.release()
    return vid


def make_gif_from_images(img_paths, outgif_path):
    import imageio
    resize_ratio = 4
    skip_ratio = 2

    with imageio.get_writer(outgif_path, mode='I') as writer:
        for img_id, img_path in enumerate(img_paths):
            image = imageio.imread(img_path)
            image_resize = image[::resize_ratio, ::resize_ratio, :]
            # Do sth to make gif file smaller
            # 1) change resolution
            # 2) change framerate
            if img_id % skip_ratio == 0:
                writer.append_data(image_resize)
    print("Gif made!")
    return
