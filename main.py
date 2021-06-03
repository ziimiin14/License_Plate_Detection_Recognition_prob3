#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:49:57 2019

@author: xingyu
"""
import sys

from numpy.core.fromnumeric import transpose
sys.path.append('./LPRNet')
sys.path.append('./yolov5')
from LPRNet_Test import *

import numpy as np
import argparse
import torch
import time
import cv2
import torch.backends.cudnn as cudnn
import warnings

from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLOv5 & LPR')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument("--scale", dest='scale', help="scale the iamge", default=1, type=int)
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    args = parser.parse_args()

    # Determine cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create LPRNet instance (License Plate Recognition)
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('LPRNet/weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
    lprnet.eval()

    print("Successful to build LPR network!")
    
    # Create STNet instance  (Spatial Transformer)
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('LPRNet/weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
    STN.eval()
    
    print("Successful to build ST network!")


    # Declare source, weights, and desired input image size for YOLOv5 network
    source, weights, imgsz = args.source, args.weights, args.img_size

    # Load YOLOv5 model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    half = True

    # Convert into fp16 and inference mode
    if half:
        model.half()
        model.eval()
    print("Successful to build YOLOv5 network!")

    # Load image folder
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    # Run inference for YOLOv5 network (detect the license car plate)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    for path, img, im0s, vid_cap in dataset:
        print(path)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=args.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms,
                                   max_det=args.max_det)
        t2 = time_synchronized()

        for i ,det in enumerate(pred):
            p, s, im0 = path, '', im0s.copy()
            
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                # Obtain top left and bottom right (x,y) positions of the detected box
                x1,y1,x2,y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])

                # Compute width and height of the detected box
                w = int(x2-x1 + 1.0)
                h = int(y2-y1 + 1.0)

                # Slice original image and obtain the detected license plate only
                img_box = np.zeros((h,w,3))
                img_box = im0s[y1:y2+1,x1:x2+1,:]

                # Resize into 94x24 for LPRNet inference purpose
                im = cv2.resize(img_box,(94,24),interpolation=cv2.INTER_CUBIC)
                im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
                data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 

                # Inference for ST network and LPR netowkr
                transfer = STN(data)
                preds = lprnet(transfer)
                preds = preds.cpu().detach().numpy()  # (1, 68, 18)

                # Decode the labels    
                labels, pred_labels = decode(preds, CHARS)


                # Draw the detected box and add license plate number into the original image
                cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 0, 255), 3)
                im0s = cv2ImgAddText(im0s, labels[0], (x1, y1-12), textColor=(255, 255, 0), textSize=25)

        im0s = cv2.resize(im0s, (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC)
        index = path[::-1].find('/')
        saved_name = path[-index:]
        cv2.imwrite('./test_result/detected_'+saved_name, im0s)
        cv2.imshow('detected_'+saved_name, im0s)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()



    
