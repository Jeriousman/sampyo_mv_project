import sys
print(sys.prefix)
import os
import io
import time
import json
import base64
import random
import socket
import pyodbc
import datetime
from datetime import timedelta
from multiprocessing import Process, Queue
import traceback
import requests
from requests.auth import HTTPDigestAuth
import statistics
import logging.handlers
import matplotlib.pyplot as plt
from PIL import Image

import timm
import torch
import pyodbc
import numpy as np
import pandas as pd
import cv2 
from onvif import ONVIFCamera
from torchvision import transforms


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from arena_api.system import system
import subprocess


#### This is for depth camera
import depthai as dai
from datetime import timedelta

import pickle
sys.path.append("/home/sdt/Workspace/onvif/python-onvif-zeep/socket")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import mssql

#### for MaskDINO #####

import multiprocessing as mp
###MV 입도분석 모델 loading 
class granule_sam:
    def __init__(self):

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = 'sam2.1_hiera_large'
        self.mixed = True
        self.checkpoint = f"/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        # self.model_cfg = "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/granule_analysis/sam2_hiera_l.yaml"
        # self.checkpoint = "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/granule_analysis/checkpoints/sam2.1_hiera_large.pt"

        self.cfg = {
            "file_path":"100.jpg",
            "points_per_side":40,
            "pred_iou_thresh":0.86,
            "stability_score_thresh":0.9,
            "crop_n_layers":2,
            "crop_n_points_downscale_factor":1,
            "box_nms_thresh":0.8,
            "min_mask_region_area":10
        }
        self.inference_result_path = '/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/sam_img'

        sam2 = build_sam2(self.model_cfg, self.checkpoint, device = self.DEVICE, apply_postprocessing=False)
        ## For faster inference, we use torch.compile
        self.sam2 = torch.compile(sam2, mode='reduce-overhead')

    def forward_granule(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = np.array(img)

        print(f'granule image shape: {img.shape}')
        img_w, img_h, img_c = img.shape
        # img_w, img_h = img.shape
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # mask generator load
        if self.mixed:
            # result_path = f'result/gs_test_{model_name}_mixed.jpg'
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                mask_generator_SAM = SAM2AutomaticMaskGenerator(
                    model = self.sam2,
                        points_per_side = self.cfg['points_per_side'],
                        pred_iou_thresh=self.cfg['pred_iou_thresh'],
                        stability_score_thresh=self.cfg['stability_score_thresh'],
                        crop_n_layers=self.cfg['crop_n_layers'],
                        crop_n_points_downscale_factor=self.cfg['crop_n_points_downscale_factor'],
                        box_nms_thresh=self.cfg['box_nms_thresh'],
                        min_mask_region_area=self.cfg['min_mask_region_area'],
                    )
                # SAM inference
                try:
                    result_SAM = mask_generator_SAM.generate(img)
                except Exception as e:
                    print(e)
        else:
            # result_path = f'result/gs_test_{model_name}.jpg'
            mask_generator_SAM = SAM2AutomaticMaskGenerator(
                model = self.sam2,
                    points_per_side = self.cfg['points_per_side'],
                    pred_iou_thresh=self.cfg['pred_iou_thresh'],
                    stability_score_thresh=self.cfg['stability_score_thresh'],
                    crop_n_layers=self.cfg['crop_n_layers'],
                    crop_n_points_downscale_factor=self.cfg['crop_n_points_downscale_factor'],
                    box_nms_thresh=self.cfg['box_nms_thresh'],
                    min_mask_region_area=self.cfg['min_mask_region_area'],
                )
            # SAM inference
            try:
                result_SAM = mask_generator_SAM.generate(img)
            except Exception as e:
                print(e)     
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print(f"inference_time_SAM : {start.elapsed_time(end)/(10**3)} sec")

        # SAM segmentation
        # cumulated_SAM= np.zeros(result_SAM[1]["segmentation"].shape)
        shape = result_SAM[0]['segmentation'].shape
        result_image = np.zeros(shape)
        #real_size = ()**2
        count = 0
        counts_SAM = 0
        start = time.time()
        for n, r in enumerate(result_SAM):
            if r['area'] < 0.1 * img_w * img_h and r['stability_score'] > self.cfg['stability_score_thresh']:
                if np.amax(result_image + r['segmentation'].astype(int)) < 2:
                # if np.amax(result_image + r['segmentation'].astype(int)) < 10:
                    result_image = result_image + r['segmentation'].astype(int)
                    count += 1
                counts_SAM += 1

        # np.amax(result_image)
        # 시각화
        
        # plt.imshow(cumulated_SAM)
        #plt.title(f'inference_time: {inference_time_SAM:.2f}sec \n count: {counts_SAM}', fontsize=16)

        # plt.figure(figsize=(6,6))
        # plt.imshow(result_image)
        plt.imsave(f"/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/enhanced_image_0_result.jpg", result_image)
        # cv2.imwrite(self.inference_result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)) 
        
        return result_image

        ##다연님이 보내준 겹치지 않게 화긴하는 방법 
        # for n, r in enumerate(result):
        #     if r['area'] < self.cfg['area_thresh'] * shape[0] * shape[1] and r['stability_score'] > self.cfg['stability_score_thresh']:
        #         if np.amax(result_image + r['segmentation'].astype(int)) < 2:
        #             result_image = result_image + r['segmentation'].astype(int)
        #             count += 1
        #             convert_mm2 = (np.sum(r['segmentation'])*(real_size**2))/10**6
        #             sizes.append(round(convert_mm2,2))


    def analize_granule(self, image):

        mm13 = 0
        mm20 = 0
        mm25 = 0
        mm40 = 0

        # image = cv2.imread('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/sampyo_sam2.jpg')
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the mask (predefined threshold)
        mask = cv2.inRange(gray, 200, 255)

        #Find all Objects contour's 
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        granule_num_px_list = []
        granule_analysis_dict = {}
        for i in range(len(contours)):
            m=max(contours[i][0])
            print(area)
            area = cv2.contourArea(contours[i])
            # print(f"Area{i+1} = {area} pixels")
            image = cv2.drawContours(image, contours, i, (0,255,255*i), -1)
            img_text = cv2.putText(image, str(area), m, cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1, cv2.LINE_AA, False)


            if area < 1450:
                mm13+=1 
            elif area >= 1700 and area < 2100:
                mm20+=1
            elif area >= 2300 and area < 2800:
                mm25+=1
            elif area >= 3000:
                mm40+=1
            granule_num_px_list.append(area)

        min_value = min(granule_num_px_list) #0
        max_value = max(granule_num_px_list) #6201
        average_value = int(np.round(sum(granule_num_px_list)/len(granule_num_px_list))) ## average ##855
        median_value = statistics.median(granule_num_px_list) ## 603.5
        stdev_value = statistics.stdev(granule_num_px_list)
        num_granule = len(contours) ##1691
        # median_value/25 ##1mm당 pixel 갯수 
        # 94.32*13
        # 94.32*20
        # 94.32*25
        # 94.32*40
        granule_analysis_dict = {"13mm": mm13, '20mm': mm20, '25mm': mm25, '40mm': mm40}

        return granule_analysis_dict, min_value, max_value, average_value, median_value, num_granule, stdev_value

granule_model = granule_sam()
img_path = '/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/enhanced_image_0.jpg'
sam_result_img = granule_model.forward_granule(img_path)
