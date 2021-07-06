# Self-Supervised Small Soccer Player Detection and Tracking

Code for the paper "Self-Supervised Small Soccer Player Detection and Tracking", published at 3rd International ACM Workshop on Multimedia Content Analysis in Sports 2020. 
Content will be soon available. 

Link of the video presentation : https://drive.google.com/file/d/1rbRKTuIOstUG4FDl0Vem3g3mT8FR9hQa/view?usp=sharing

This work contains code or parts of code taken from the following github repositories : 
* [LightTrack: A Generic Framework for Online Top-Down Human Pose Tracking](https://github.com/Guanghan/lighttrack)
* [Sports Camera Calibration via Synthetic Data](https://github.com/lood339/pytorch-two-GAN)
* [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
* [Soft NMS](https://github.com/DocF/Soft-NMS)

**Prerequisites**

* Set up a Python3 environment. The code has been developed in python 3.7.  
* Install packages given in requirement.txt 

## Testing 

**Download data**

* Download evaluation datasets (frames and annotations) "issia" / "SPD" / "TV_soccer" at this [google drive link](https://drive.google.com/drive/folders/1dE1yzHyBOVGs4A1VlmFTq_TXOT1S5f_b?usp=sharing) and extract them in the data folder

**Player detection**

Download the player detection models and extract them in the checkpoints_runs folder 
* [Fine-tuned Resnet50 teacher model](https://drive.google.com/file/d/1ewjgLM7BHpFv1fAhKCX-wKN2otuFU7Kr/view?usp=sharing) 
* [Fine-tuned Resnet18 student model](https://drive.google.com/file/d/1_umt5UvyF-XZCVfyNSiugheDzgtviiag/view?usp=sharing)

The script eval_fasterRCNN.py enables to get the mAP score of the model on the dataset of your choice and to save the images along with detected player boxes.
To save the images use the option '--save_visualization'. Images will be saved in the folder 'script/detection/results'
* The command below gives an exemple to evaluate and visualize detection with the Resnet50 teacher model on the TV_soccer evauation dataset. 
```
cd script/detection
python eval_fasterRCNN.py --backbone resnet50 --test_dataset_name TV_soccer --save_visualization --checkpoint ../../checkpoints_runs/player_det_resnet50_teacher.pth
```
* The command below gives an exemple to evaluate and visualize detection with the Resnet18 student model on the SPD evauation dataset. 
```
cd script/detection
python create_dataset.py --backbone resnet18 --test_dataset_name SPD --save_visualization --checkpoint ../../checkpoints_runs/player_det_resnet18_student.pth --use_context
```
* In order down-scale (and pad) images by a certain factor to work with smaller player, use the command '--scale_transform_test factor'

**Player tracking**

### Soccer Player Detection

## Testing 

## Training 

**Download data**

If you need to realize training, download the training dataset (frames and annotations) "SoccerNet" at this [google drive link](https://drive.google.com/drive/folders/1dE1yzHyBOVGs4A1VlmFTq_TXOT1S5f_b?usp=sharing) and extract it in the data folder

**Automatic self-labeling of the training image with the teacher network**

The code for automatic self-labeling of training data is given in script/automatic_annotation. 
This process is very long to run on the full SoccerNet dataset. If you want to avoid it, we give the result of this automatic annotation in at this [google drive link](https://drive.google.com/drive/folders/1ZFIxtyNjyBov2z2izNQHGP3k-ECwcOpu?usp=sharing) in the file "annoration_r1.tar.xz". 

If you are interesting in this annotation process, you can also try and evaluate it on the evaluation datasets. To do so : 

* Clone the [lighttrack](https://github.com/Guanghan/lighttrack) repository in script/other_utils/. And download the corresponding model.
```
cd script/other_utils
git clone https://github.com/Guanghan/lighttrack.git
bash ./download_weights.sh
```
* Download the GAN pix2pix line detection model at [google drive link](https://drive.google.com/drive/folders/1H-zIEBe-gJtJe-y_XztfujJ6VZOB9799?usp=sharing) and extract it in checkpoints_runs.
* The command below gives an exemple to evaluate the annotation process on the TV_soccer dataset.
```
cd script/automatic_annotation
python create_dataset.py --data_name TV_soccer
```
Note that field detection and player detection is runned once and for all and saved for each dataset, the second time you call the code in a dataset, with the same hyper-parameters, field detection and player detection will not be computed again.
* The command below enables to extract annotations on the non-annotated SoccerNet training data. 
```
cd script/automatic_annotation
python create_dataset.py --data_name SoccerNet --create_data
```

**Fine-tuning of the teacher network**

**Training of the student network**

### Player Tracking

