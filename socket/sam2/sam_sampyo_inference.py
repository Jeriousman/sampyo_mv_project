

import pickle
import csv
import json
import time
import random
import argparse
import cv2
import numpy as np
# import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm
from datetime import datetime

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# import segment_anything
from torch import nn
#from super_image import EdsrModel, ImageLoader
import matplotlib.pyplot as plt
from PIL import Image
import requests
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import statistics

print(torch.cuda.is_available())
# Set Variables


# model_name = "sam2.1_hiera_base_plus"
# model_name = "sam2.1_hiera_tiny"
# model_name = "sam2.1_hiera_small"
# model_name = "sam2.1_hiera_large"

# model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
# model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

class granule_sam:
    def __init__(self):

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = 'sam2.1_hiera_large'
        self.mixed = True
        self.checkpoint = f"./checkpoints/{self.model_name}.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
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
        self.inference_result_path = '/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/granule_test'

        sam2 = build_sam2(self.model_cfg, self.checkpoint, device = self.DEVICE, apply_postprocessing=False)
        ## For faster inference, we use torch.compile
        self.sam2 = torch.compile(sam2, mode='reduce-overhead')

    def forward_granule(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_w, img_h, img_c = img.shape
        print(f'image shape: {img.shape}')
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
        # plt.figure(figsize=(6,6))
        # plt.imshow(cumulated_SAM)
        #plt.title(f'inference_time: {inference_time_SAM:.2f}sec \n count: {counts_SAM}', fontsize=16)
        # plt.imshow(result_image)
        # cv2.imwrite(self.inference_result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)) 
        plt.imsave(f"{self.inference_result_path}/{img_path.split('/')[-1]}", result_image)

        return result_image

        ##다연님이 보내준 겹치지 않게 화긴하는 방법 
        # for n, r in enumerate(result):
        #     if r['area'] < self.cfg['area_thresh'] * shape[0] * shape[1] and r['stability_score'] > self.cfg['stability_score_thresh']:
        #         if np.amax(result_image + r['segmentation'].astype(int)) < 2:
        #             result_image = result_image + r['segmentation'].astype(int)
        #             count += 1
        #             convert_mm2 = (np.sum(r['segmentation'])*(real_size**2))/10**6
        #             sizes.append(round(convert_mm2,2))


    def analyze_granule(self, image):

        mm13 = 0
        mm20 = 0
        mm25 = 0
        mm40 = 0
        # print(type(image))
        # image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the mask (predefined threshold)
        mask = cv2.inRange(image, 1, 255)

        #Find all Objects contour's 
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        granule_num_px_list = []
        granule_analysis_dict = {}
        for i in range(len(contours)):
            m=max(contours[i][0])
            area = cv2.contourArea(contours[i])
            # print(f"Area{i+1} = {area} pixels")
            image = cv2.drawContours(image, contours, i, (0,255,255*i), -1)
            img_text = cv2.putText(image, str(area), m, cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1, cv2.LINE_AA, False)



            if area > 700 and area < 1400:
                mm13+=1 
            elif area >= 1400 and area < 2050:
                mm20+=1
            elif area >= 2100 and area < 3500:
                mm25+=1
            elif area > 6000 and area <= 6500 :
                mm40+=1

            # if area > 700 and area < 1400:
            #     mm13+=1 
            # elif area >= 1400 and area < 2050:
            #     mm20+=1
            # elif area >= 2100 and area < 3000:
            #     mm25+=1
            # elif area > 3800 and area <= 4200 :
            #     mm40+=1
            granule_num_px_list.append(area)

        


        # min_value = min(granule_num_px_list) #0
        # max_value = max(granule_num_px_list) #6201
        # average_value = int(np.round(sum(granule_num_px_list)/len(granule_num_px_list))) ## average ##855
        # median_value = statistics.median(granule_num_px_list) ## 603.5
        # # stdev_value = statistics.stdev(granule_num_px_list)
        # num_granule = len(contours) ##1691

        granule_analysis_dict = {"13mm": int(mm13), '20mm': int(mm20), '25mm': int(mm25), '40mm': int(mm40)}
        return granule_analysis_dict, granule_num_px_list
        return min_value, max_value, average_value, median_value, num_granule, int(mm13), int(mm20), int(mm25), int(mm40)
        # median_value/25 ##1mm당 pixel 갯수 
        # 94.32*13
        # 94.32*20
        # 94.32*25
        # 94.32*40
        # granule_analysis_dict = {"13mm": mm13, '20mm': mm20, '25mm': mm25, '40mm': mm40}



sam_model = granule_sam()

analysis_list = []
for img_path in [
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-13 09:48:17.826993_202502130019.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-12 10:25:18.704127_202502120028.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-12 10:28:43.833396_202502120029.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-12 11:00:36.630675_202502120035.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-12 13:05:16.777595_202502120048.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-12 13:11:26.572062_202502120050.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-12 13:38:17.584715_202502120056.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-12 14:54:07.415732_202502120066.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-13 09:03:45.368707_202502130010.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-13 12:00:18.751012_202502130033.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-13 14:38:20.856462_202502130048.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-13 16:04:33.051312_202502130058.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-14 09:01:08.011588_202502140013.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-14 10:01:46.551011_202502140025.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-14 10:27:05.165147_202502140030.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-14 10:46:30.252772_202502140035.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-14 11:57:35.361846_202502140041.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-14 14:49:42.800664_202502140056.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-14 15:03:33.629197_202502140057.jpg"]:
    print(img_path)
    sam_result_image = sam_model.forward_granule(img_path)
    analysis_result = sam_model.analyze_granule(sam_result_image)
    
    analysis_list.append(analysis_result)

img_path = "/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-14 11:57:35.361846_202502140041.jpg"
sam_result_image = sam_model.forward_granule(img_path)
analysis_result = sam_model.analyze_granule(sam_result_image)

import pickle
with open("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/mv_analysis.pickle", "wb") as f:
    pickle.dump(analysis_list, f)

medians = []
for result in analysis_list:
    medians.append(result[3])

np.average(medians)


##102.2 per pixel

##25mm
# 2427 2555 2682 ##5%
# 2299 2555 2810 ##10%


##20mm
##1941 2044 2146 5%
##1839 2044 2248 10%


##13mm
# 1262 1328 1394
# 




sam_model = granule_sam()

analysis_list = []
granule_num_px_list = []
for img_path in [
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:26:18.646082_202502270083.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:16:20.027107_202502270079.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:10:01.044576_202502270076.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 12:21:37.797464_202502270073.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 12:16:09.113308_202502270072.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 12:15:52.824708_202502270071.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 10:57:12.383111_202502270053.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 10:47:49.571115_202502270050.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 10:43:10.015462_202502270048.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 12:15:52.824708_202502270071.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 12:16:09.113308_202502270072.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:10:01.044576_202502270076.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:16:20.027107_202502270079.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:26:18.646082_202502270083.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:41:41.751077_202502270086.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:44:36.008223_202502270087.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 14:35:17.997702_202502270099.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 14:28:42.938776_202502270097.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 14:19:31.555614_202502270095.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 14:13:04.869355_202502270093.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:52:57.336360_202502270088.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 13:56:27.896068_202502270090.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 14:49:31.210350_202502270101.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-27 15:02:08.643570_202502270103.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-06 14:29:56.189714_202503060113.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-06 14:25:28.203044_202503060111.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-06 14:21:18.709647_202503060110.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-06 14:17:58.924291_202503060109.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-05 15:50:43.625287_202503050105.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-05 15:12:13.402821_202503050099.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-05 14:50:17.543367_202503050094.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-05 14:47:13.048282_202503050093.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-05 14:38:42.169886_202503050091.jpg",
"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-03-05 14:23:14.128766_202503050085.jpg",
]:
    print(img_path)
    sam_result_image = sam_model.forward_granule(img_path)
    analysis_result, granule_num_px_result= sam_model.analyze_granule(sam_result_image)
    
    analysis_list.append(analysis_result)
    granule_num_px_list.append(granule_num_px_result)
    
len(granule_num_px_list)

flattened_granule_num_px_list = []

for i in granule_num_px_list:
    flattened_granule_num_px_list.extend(i)
len(flattened_granule_num_px_list)


flattened_granule_num_px_list

import statistics

statistics.mean(flattened_granule_num_px_list)

average_value = statistics.mean(flattened_granule_num_px_list) ##3259.2635113003603
median_value = statistics.median(flattened_granule_num_px_list) ##2872.5
min_value = min(flattened_granule_num_px_list) ##0
max_value = max(flattened_granule_num_px_list) ##34888.5
std_value = statistics.stdev(flattened_granule_num_px_list) ##1993.9732696457459

mm13 = 0
mm20 = 0
mm25 = 0
mm40 = 0

for img in analysis_list:
    mm13 += img['13mm']
    mm20 += img['20mm']
    mm25 += img['25mm']
    mm40 += img['40mm']


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4)
size = ['13mm', '20mm', '25mm', '40mm']
values = [mm13, mm20, mm25, mm40]

plt.bar(x, values)
plt.xticks(x, size)

plt.show()


#with open('/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/flattened_granule_num_px_list.pkl', 'wb') as f:
#    pickle.dump(flattened_granule_num_px_list, f)

#with open('/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/analysis_list.pkl', 'wb') as f:
#    pickle.dump(analysis_list, f)

with open('/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/flattened_granule_num_px_list.pkl', 'rb') as f:
    flattened_granule_num_px_list = pickle.load(f)

with open('/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/analysis_list.pkl', "rb") as f:
    analysis_list = pickle.load(f)



np.sort(flattened_granule_num_px_list)[:100]
max(flattened_granule_num_px_list)


# print(type(image))
z_values = []
for i in flattened_granule_num_px_list:
    z = (i - 3259.2635113003603)/1993.9732696457459
    z_values.append(z)
    print(z)

len(z_values)

z_values_wo_outliers = [x for x in z_values if -2 <= x <= 2]

##outlier popped
#for i in z_values:
#    print(i)
#    if i > 2 or i < -2:
#        z_values.pop()


datapoints_wo_outlier = []
for i in z_values_wo_outliers:
    datapoint_wo_outlier = i*1993.9732696457459 + 3259.2635113003603
    datapoints_wo_outlier.append(datapoint_wo_outlier)
    print(datapoint_wo_outlier)







average_value = statistics.mean(datapoints_wo_outlier) ##2997.762821974014
median_value = statistics.median(datapoints_wo_outlier) ##2786.0
min_value = min(datapoints_wo_outlier) ##0
max_value = max(datapoints_wo_outlier) ##7241.5
std_value = statistics.stdev(datapoints_wo_outlier) ##1511.4829086136497

mean_v = average_value/25 ## 164.84 -> 1 mm = 119.91 pixels 
#boundary_value = mean_v*0.10
#mean_v = median_value/25 ## or...  using original mean value,  130.36
boundary_value = mean_v*0.05















##1
mm13 = 0
mm20 = 0
mm25 = 0
mm40 = 0
for i in datapoints_wo_outlier:
    if i > 1919.268 and i < 2345:
        mm13+=1 
    elif i >= 2952.72 and i < 3608.88:
        mm20+=1
    elif i >= 2507 and i < 3297:
        mm25+=1
    elif i > 5905.44 and i <= 7217.76:
        mm40+=1




4101*1.1
4101*0.9


##2
mm40_ = mean_v * 40

mm40_high = mm40_*1.1
mm40_low = mm40_*0.9


mm25_ = mean_v * 25
mm25_high = mm25_*1.1
mm25_low = mm25_*0.9

mm20_ = mean_v * 20
mm20_high = mm20_*1.1
mm20_low = mm20_*0.9

mm13_ = mean_v * 13
mm13_high = mm13_*1.1
mm13_low = mm13_*0.9



##3  ##The best
mm40_ = mean_v * 40
range_value = 300
mm40_high = mm40_ + range_value
mm40_low = mm40_ - range_value


mm25_ = mean_v * 25
mm25_high = mm25_ + range_value
mm25_low = mm25_ - range_value

mm20_ = mean_v * 20
mm20_high = mm20_ + range_value
mm20_low = mm20_ - range_value

mm13_ = mean_v * 13
mm13_high = mm13_ + range_value
mm13_low = mm13_ - range_value




mm13 = 0
mm20 = 0
mm25 = 0
mm40 = 0
for i in datapoints_wo_outlier:
    if i > mm13_low and i < mm13_high:
        mm13+=1 
    elif i >= mm20_low and i < mm20_high:
        mm20+=1
    elif i >= mm25_low and i < mm25_high:
        mm25+=1
    elif i > mm40_low and i <= mm40_high:
        mm40+=1






##4
mm13 = 0
mm20 = 0
mm25 = 0
mm40 = 0


for i in datapoints_wo_outlier:
    #print(i)
    if i > mm13_ - boundary_value and i < mm13_ +boundary_value:
        mm13+=1 
    elif i >= mm20_ -boundary_value and i < mm20_ +boundary_value:
        mm20+=1
    elif i >= mm25_ -boundary_value and i < mm25_ + boundary_value:
        mm25+=1
    elif i > mm40_-boundary_value and i <= mm40_ +boundary_value:
        mm40+=1





import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4)
size = ['13mm', '20mm', '25mm', '40mm']
values = [mm13, mm20, mm25, mm40]

plt.bar(x, values)
plt.xticks(x, size)
plt.title('Granule analysis result')
plt.xlabel('granule classes')
plt.ylabel('No. of granule by classes')

plt.show()
