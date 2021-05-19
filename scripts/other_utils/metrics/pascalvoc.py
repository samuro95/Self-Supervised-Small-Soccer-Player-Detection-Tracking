#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

"""
MODIFED FROM https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/pascalvoc.py

Copyright (c) 2018 Rafael Padilla
"""


import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from metric_utils import BBFormat
import numpy as np


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses



def compute_metrics(gtFolder, detFolder, iouThreshold = 0.5, gtFormat = 'xyrb', detFormat = 'xyrb',
            gtCoordinates = 'abs', detCoordinates = 'abs', showPlot = False, savePath = None) :

            currentPath = os.path.dirname(os.path.abspath(__file__))

            # Arguments validation
            errors = []
            # Validate formats
            gtFormat = ValidateFormats(gtFormat, '-gtformat', errors)
            detFormat = ValidateFormats(detFormat, '-detformat', errors)


            # Coordinates types
            gtCoordType = ValidateCoordinatesTypes(gtCoordinates, '-gtCoordinates', errors)
            detCoordType = ValidateCoordinatesTypes(detCoordinates, '-detCoordinates', errors)

            imgSize = (0, 0)
            if gtCoordType == CoordinatesType.Relative:  # Image size is required
                imgSize = ValidateImageSize(imgSize, '-imgsize', '-gtCoordinates', errors)
            if detCoordType == CoordinatesType.Relative:  # Image size is required
                imgSize = ValidateImageSize(imgSize, '-imgsize', '-detCoordinates', errors)

            if savePath is not None:
                savePath = ValidatePaths(savePath, '-sp/--savepath', errors)
            else:
                savePath = os.path.join(currentPath, 'results')
            # Validate savePath
            # If error, show error messages
            if len(errors) != 0:
                print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                            [-detformat] [-save]""")
                print('Object Detection Metrics: error(s): ')
                [print(e) for e in errors]
                sys.exit()

            # Create directory to save results
            shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
            os.makedirs(savePath)
            # Show plot during execution
            showPlot = showPlot

            # print('iouThreshold= %f' % iouThreshold)
            # print('savePath = %s' % savePath)
            # print('gtFormat = %s' % gtFormat)
            # print('detFormat = %s' % detFormat)
            # print('gtFolder = %s' % gtFolder)
            # print('detFolder = %s' % detFolder)
            # print('gtCoordType = %s' % gtCoordType)
            # print('detCoordType = %s' % detCoordType)
            # print('showPlot %s' % showPlot)

            # Get groundtruth boxes
            allBoundingBoxes, allClasses = getBoundingBoxes(
                gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
            # Get detected boxes
            allBoundingBoxes, allClasses = getBoundingBoxes(
                detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
            allClasses.sort()

            evaluator = Evaluator()
            acc_AP = 0
            validClasses = 0

            # Plot Precision x Recall curve
            detections = evaluator.PlotPrecisionRecallCurve(
                allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
                IOUThreshold=iouThreshold,  # IOU threshold
                method=MethodAveragePrecision.EveryPointInterpolation,
                showAP=True,  # Show Average Precision in the title of the plot
                showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
                savePath=savePath,
                showGraphic=showPlot)

            f = open(os.path.join(savePath, 'results.txt'), 'w')
            f.write('Object Detection Metrics\n')
            f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
            f.write('Average Precision (AP), Precision and Recall per class:')

            # each detection is a class
            for metricsPerClass in detections:

                # Get metric values per each class
                cl = metricsPerClass['class']
                ap = metricsPerClass['AP']
                precision = np.array(metricsPerClass['precision'])
                recall = np.array(metricsPerClass['recall'])
                totalPositives = metricsPerClass['total positives']
                total_TP = metricsPerClass['total TP']
                total_FP = metricsPerClass['total FP']

                if totalPositives > 0:
                    validClasses = validClasses + 1
                    acc_AP = acc_AP + ap
                    prec = ['%.2f' % p for p in precision]
                    rec = ['%.2f' % r for r in recall]
                    ap_str = "{0:.2f}%".format(ap * 100)
                    # ap_str = "{0:.4f}%".format(ap * 100)
                    print('AP: %s (%s)' % (ap_str, cl))
                    f.write('\n\nClass: %s' % cl)
                    f.write('\nAP: %s' % ap_str)
                    f.write('\nPrecision: %s' % prec)
                    f.write('\nRecall: %s' % rec)


            # print(recall)
            # print(precision)

            precision = precision[-1]
            recall = recall[-1]

            print(recall)
            print(precision)
            f1 = 2*precision*recall/(precision+recall)
            print('mean f1 ', f1)
            mAP = acc_AP / validClasses
            mAP_str = "{0:.2f}%".format(mAP * 100)
            print('mAP: %s' % mAP_str)
            f.write('\n\n\nmAP: %s' % mAP_str)


            return(mAP)


if __name__== '__main__' :

    # Get current path to set default folders
    currentPath = os.path.dirname(os.path.abspath(__file__))

    VERSION = '0.1 (beta)'

    parser = argparse.ArgumentParser(
        prog='Object Detection Metrics - Pascal VOC',
        description='This project applies the most popular metrics used to evaluate object detection '
        'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
        'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
    # formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    # Positional arguments
    # Mandatory
    parser.add_argument(
        '-gt',
        '--gtfolder',
        dest='gtFolder',
        default=os.path.join(currentPath, 'groundtruths'),
        metavar='',
        help='folder containing your ground truth bounding boxes')
    parser.add_argument(
        '-det',
        '--detfolder',
        dest='detFolder',
        default=os.path.join(currentPath, 'detections'),
        metavar='',
        help='folder containing your detected bounding boxes')
    # Optional
    parser.add_argument(
        '-t',
        '--threshold',
        dest='iouThreshold',
        type=float,
        default=0.5,
        metavar='',
        help='IOU threshold. Default 0.5')
    parser.add_argument(
        '-gtformat',
        dest='gtFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the ground truth bounding boxes: '
        '(\'xywh\': <left> <top> <width> <height>)'
        ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-detformat',
        dest='detFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the detected bounding boxes '
        '(\'xywh\': <left> <top> <width> <height>) '
        'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-gtcoords',
        dest='gtCoordinates',
        default='abs',
        metavar='',
        help='reference of the ground truth bounding box coordinates: absolute '
        'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-detcoords',
        default='abs',
        dest='detCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: '
        'absolute values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-imgsize',
        dest='imgSize',
        metavar='',
        help='image size. Required if -gtcoords or -detcoords are \'rel\'')
    parser.add_argument(
        '-sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
    parser.add_argument(
        '-np',
        '--noplot',
        dest='showPlot',
        action='store_false',
        help='no plot is shown during execution')
    args = parser.parse_args()


    map = compute_metrics(args.gtFolder, args.detfolder, iouThreshold = args.iouThreshold, gtFormat = args.gtFormat,
    detformat = args.detformat, gtCoordinates = args.gtCoordinates, detcoords = args.detcoords,
    showPlot = args.showPlot, savePath = args.savePath)
