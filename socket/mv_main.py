



#######################################################################################
#Lucid MV camera packages import
import cv2
import numpy as np
from arena_api.system import system
import time





#######################################################################################
#SAM2 packages import
import csv
import json
import random
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from datetime import datetime
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import requests
#import sys
#sys.path.append('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/SAM2')
import sam2
dir(sam2)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


import statistics

#######################################################################################
## SAM2 settings

## Nvidia GPU CUDA check
print(torch.cuda.is_available())
## Set Variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## Checking if device is CUDA
#print(DEVICE)

model_name = 'large2.1'
## Mixed precision (fp16 and fp32). If True, use mixed precision.
mixed = True

## Pre-trained model and cofig path set
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
#model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
#checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
#model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

## sam2 configs. It can be changed up to your taste.
cfg = {
    "points_per_side":36,
    "pred_iou_thresh":0.86,
    "stability_score_thresh":0.9,
    "crop_n_layers":2,
    "crop_n_points_downscale_factor":1,
    "box_nms_thresh":0.8,
    "min_mask_region_area":10
}

#######################################################################################
## Initializing sam2 model
sam2 = build_sam2(model_cfg, checkpoint, device =DEVICE, apply_postprocessing=False)

## For faster inference, we use torch.compile
sam2 = torch.compile(sam2, mode='reduce-overhead')




#######################################################################################
## Lucid MV camera setting
device_infos = system.device_infos
device_infos
device = system.create_device(device_infos=device_infos[0])[0]

#######################################################################################
"""
Granule analysis starting by video streaming from MV camera and SAM2
"""

## Video streaming to the MV camera
device.start_stream()


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


try:
    while True:
       ##sensor signal
       signal = input()
       if signal:

        image_buffer = device.get_buffer()
        ##ctypeslib is a module that makes C library run in python. ##pdata = pixel data?
        ##(3000, 4096, 1) shape
        np_img = np.ctypeslib.as_array(image_buffer.pdata,shape=(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))
        ##(3000, 4096, 3) shape. Grayscale to RGB 
        img = Image.open('/home/sdt/Workspace/sampyo/granule-analysis/lucid-mvcam/test_1.jpg')
        resized_img = img.resize((480,648), resample=Image.Resampling.LANCZOS)
        resized_np_img = np.array(resized_img)
        

        ##Only if you use cv2 instead of PIL
        #rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BayerRG2RGB)
        #resized_rgb_img = cv2.resize(rgb_img, (1280,1748))
        #resized_rgb_img = cv2.resize(rgb_img, (480,648))
        import cv2
        aa = cv2.imread("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/sam2_mv.png")
        aa.shape
        from PIL import Image
        import numpy as np
        img = Image.open("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/sam2_mv.png")
        resized_np_img = np.array(img)
        resized_np_img = resized_np_img[...,:3]
        resized_np_img.shape
        start.record()
        if mixed:
            result_path = f'result/gs_test_{model_name}_mixed.jpg'
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                mask_generator_SAM = SAM2AutomaticMaskGenerator(
                    model = sam2,
                        points_per_side = cfg['points_per_side'],
                        pred_iou_thresh=cfg['pred_iou_thresh'],
                        stability_score_thresh=cfg['stability_score_thresh'],
                        crop_n_layers=cfg['crop_n_layers'],
                        crop_n_points_downscale_factor=cfg['crop_n_points_downscale_factor'],
                        box_nms_thresh=cfg['box_nms_thresh'],
                        min_mask_region_area=cfg['min_mask_region_area'],
                    )
                try:
                    result_SAM = mask_generator_SAM.generate(resized_np_img)
                except Exception as e:
                    print(e)
        else:
            result_path = f'result/gs_test_{model_name}.jpg'
            mask_generator_SAM = SAM2AutomaticMaskGenerator(
                model = sam2,
                    points_per_side = cfg['points_per_side'],
                    pred_iou_thresh=cfg['pred_iou_thresh'],
                    stability_score_thresh=cfg['stability_score_thresh'],
                    crop_n_layers=cfg['crop_n_layers'],
                    crop_n_points_downscale_factor=cfg['crop_n_points_downscale_factor'],
                    box_nms_thresh=cfg['box_nms_thresh'],
                    min_mask_region_area=cfg['min_mask_region_area'],
                )
            try:
                result_SAM = mask_generator_SAM.generate(resized_np_img)
            except Exception as e:
                print(e)
        end.record()
        torch.cuda.synchronize()
        print(f"inference_time_SAM : {start.elapsed_time(end)/(10**3)} sec")
        # break


np.sum(result_SAM[0]['segmentation'])

# SAM segmentation
# start = time.time()
cumulated_SAM= np.zeros(result_SAM[1]["segmentation"].shape)
shape = result_SAM[0]['segmentation'].shape
result_image = np.zeros(shape)



np.sum(result_SAM[0]['segmentation'])
result_SAM[0]['segmentation'].shape
np.sum(result_SAM[0]['segmentation'] ==True)
np.sum(result_SAM[0]['segmentation'] ==False)


cumulated_SAM= np.zeros(result_SAM[1]["segmentation"].shape)
shape = result_SAM[0]['segmentation'].shape
result_image = np.zeros(shape)
result_SAM['area']

result_SAM[0].keys()


#real_size = ()**2
count = 0
sizes = []
objects = []
counts_SAM = 0
start = time.time()
granule_analysis_dict = {}
granule_analysis_list = []
## iterate over all the instances we predicted.
for n, instance in enumerate(result_SAM):
    if instance['stability_score'] > 0.9:
        granule_analysis_dict[n] = {'num_px': instance['area'], 'score': instance['stability_score']}
        granule_analysis_list.append(instance['area'])
        result_image = result_image + instance['segmentation'].astype(int)
        min(granule_analysis_list)
        max(granule_analysis_list)
        sum(granule_analysis_list)/len(granule_analysis_list) ## average
        # statistics.median(granule_analysis_list)
        ## 15, 17 21, 25mm





plt.imshow(result_image)
plt.imsave('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/sam2_mv_result.jpg', result_image)




    instance['segmentation']
    ## instance['area']: number of pixels of each instance. same value with "np.sum(r['segmentation'].astype(int))"
    instance['area']
    ## bounding box of instance detected
    instance['bbox']
    ## predicted_iou?
    instance['predicted_iou']
    ## point_coords??
    instance['point_coords']
    instance['stability_score']
    ## ?
    instance['crop_box']
    



    if r['area'] < 0.1 * 1280 * 1024 and r['stability_score'] > cfg['stability_score_thresh']:
        if np.amax(result_image + r['segmentation'].astype(int)) < 2:
                result_image = result_image + r['segmentation'].astype(int)
                # plt.imsave(f'result/gs_test_large_mixed_{n}.jpg', result_image) ##모든 세그멘트들이 누적되어 표현되는 것을 확인
                #objects.append(r['segmentation'].astype(int))
                count += 1
                #sizes.append(r['area'])
                #sizes.append(round(np.sum(r['segmentation'])*(real_size**2),2))
        # cumulated_SAM = cumulated_SAM + r['segmentation']
        # x, y, w, h = r['bbox']
        # sizes.append(r['area'])
        #sizes.append(np.sum(r['segmentation'])*21.826)
        # cumulated = cumulated + r['segmentation'].astype(np.uint8)*255
        counts_SAM += 1
# end = time.time()
# 시각화
#plt.figure(figsize=(6,6))
#plt.imshow(cumulated_SAM)
#plt.title(f'inference_time: {inference_time_SAM:.2f}sec \n count: {counts_SAM}', fontsize=16)
#done = time.time()






        #cv2.imshow('lucid_img', rgb_img)
        #cv2.imwrite('inputtest.jpg', rgb_img)
        device.requeue_buffer(image_buffer)
        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()
            device.stop_stream()

        

        
except Exception as e:
    print(e)
    device.stop_stream()









np.sum(result_SAM[0]['segmentation'])

# SAM segmentation
# start = time.time()
cumulated_SAM= np.zeros(result_SAM[1]["segmentation"].shape)
shape = result_SAM[0]['segmentation'].shape
result_image = np.zeros(shape)
#real_size = ()**2
count = 0
sizes = []
objects = []
counts_SAM = 0
start = time.time()
for n, r in enumerate(result_SAM):
    if r['area'] < 0.1 * 1280 * 1024 and r['stability_score'] > cfg['stability_score_thresh']:
        if np.amax(result_image + r['segmentation'].astype(int)) < 2:
                result_image = result_image + r['segmentation'].astype(int)
                # plt.imsave(f'result/gs_test_large_mixed_{n}.jpg', result_image) ##모든 세그멘트들이 누적되어 표현되는 것을 확인
                #objects.append(r['segmentation'].astype(int))
                count += 1
                #sizes.append(r['area'])
                #sizes.append(round(np.sum(r['segmentation'])*(real_size**2),2))
        # cumulated_SAM = cumulated_SAM + r['segmentation']
        # x, y, w, h = r['bbox']
        # sizes.append(r['area'])
        #sizes.append(np.sum(r['segmentation'])*21.826)
        # cumulated = cumulated + r['segmentation'].astype(np.uint8)*255
        counts_SAM += 1
# end = time.time()
# 시각화
#plt.figure(figsize=(6,6))
#plt.imshow(cumulated_SAM)
#plt.title(f'inference_time: {inference_time_SAM:.2f}sec \n count: {counts_SAM}', fontsize=16)
#done = time.time()




# sorted_array_list = sorted(objects, key=lambda x: np.sum(x), reverse=False)

# from matplotlib.widgets import Slider


# slider = widgets.IntSlider(min=0, max=len(array_list) - 1, step=1, description='Index')

# # 슬라이더에 따라 이미지를 업데이트하는 함수
# def update_image(idx):
#     plt.imshow(array_list[idx], cmap='gray')
#     plt.show()

# # 슬라이더와 콜백 함수 연결
# widgets.interact(update_image, idx=slider)

# result_SAM[0]

# plt.imshow(cumulated_SAM)


# def convert_to_polygon(segmentation):
#     # 세그멘테이션 정보를 바이너리 이미지로 변환
#     binary_mask = (segmentation > 0.5).astype(np.uint8)

#     # 윤곽선 추출
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Cityscapes 형식의 polygon으로 변환
#     polygons = []
#     for contour in contours:
#         polygon = contour.flatten().tolist()
#         polygons.append(polygon)

#     return polygons

# # SAM 모델의 세그멘테이션 정보 (예시, 2D NumPy 배열)
# sam_segmentation = result_SAM[0]['segmentation']
# result = convert_to_polygon(sam_segmentation)

# print(result)


# from matplotlib.patches import Polygon
# # 이미지 경로 및 세그멘테이션 좌표 가져오기 (예시에서는 첫 번째 이미지 사용)
# # image_info = annotations['images'][0]
# # image_path = 'path/to/your/images/' + image_info['file_name']
# segmentations = cityscapes_annotation['objects'][0]['polygon']

# # 이미지 불러오기
# # image = plt.imread(image_path)

# # 시각화를 위한 설정
# fig, ax = plt.subplots(1, figsize=(8, 8))
# ax.imshow(img_rgb)

# # 세그멘테이션 좌표를 다각형으로 그리기
# for segmentation in segmentations:
#     polygon = np.array(segmentation).reshape((int(len(segmentation) / 2), 2))
#     poly = Polygon(polygon, facecolor='none', edgecolor='r')
#     ax.add_patch(poly)

# # 그림 보여주기
# plt.show()


# def inference(self, image):
#     result = self.mask_generator.generate(image)
#     shape = result[0]['segmentation'].shape
#     result_image = np.zeros(shape)
#     count = 0
#     sizes = []

#     for n, r in enumerate(result):
#         if r['area'] < self.cfg['area_thresh'] * shape[0] * shape[1] and r['stability_score'] > self.cfg['stability_score_thresh']:
#             if np.amax(result_image + r['segmentation'].astype(int)) < 2: # 겹치는 segment는 무시하기 위한 조건
#                 result_image = result_image + r['segmentation'].astype(int)
#                 count += 1
#                 x, y, w, h = r['bbox']
#                 sizes.append(np.mean([w, h])) # x, y의 평균값을 구해서 리스트에 추가

#     return result_image, count, sizes