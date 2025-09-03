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

print(torch.cuda.is_available())

# Set Variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
print(DEVICE)
# DEVICE = torch.device("cpu")

# SAM model Load
# sam = sam_model_registry['vit_h'](checkpoint = './sam_vit_h_4b8939-Copy.pth')
# sam = sam.to(DEVICE)

#model_name = 'base_plus'
model_name = 'large2.1'
#model_version = '2.1'
mixed = True
#themodel =  f"{model_name}{model_version}"
#sam2_checkpoint = f"./checkpoints/sam{model_version}_hiera_{model_name}.pt"
"""
if themodel == 'base_plus2':
    model_cfg = "sam2_hiera_b+.yaml"
elif themodel == 'large2': 
    model_cfg = "sam2_hiera_l.yaml"
elif themodel == 'small2': 
    model_cfg = "sam2_hiera_s.yaml"
elif themodel == 'tiny2': 
    model_cfg = "sam2_hiera_t.yaml"
elif themodel == 'base_plus2.1':
    model_cfg = "sam2.1_hiera_b+.yaml"
elif themodel == 'large2.1': 
    model_cfg = "sam2.1_hiera_l.yaml"
elif themodel == 'small2.1': 
    model_cfg = "sam2.1_hiera_s.yaml"
elif themodel == 'tiny2.1': 
    model_cfg = "sam2.1_hiera_t.yaml"
model_cfg = '/home/sdt/Workspace/sampyo/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml'
sam2_checkpoint = "/home/sdt/Workspace/sampyo/sam2/checkpoints/sam2.1_hiera_large.pt"
"""


#checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
#model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
#checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
#model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
##predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))



sam2 = build_sam2(model_cfg, checkpoint, device =DEVICE, apply_postprocessing=False)

# sam2 = torch.compile(sam2, mode='max-autotune')
sam2 = torch.compile(sam2, mode='reduce-overhead')

# ###dynamic quantization
# qconfig = torch.ao.quantization.get_default_qconfig('x86')
# quantized_sam2 = torch.ao.quantization.quantize_dynamic(
#     sam2, {nn.Linear}, dtype=torch.qint8
# )
# torch.save(quantized_sam2.state_dict(), "checkpoints/sam2_hiera_base_plus_dq8.pt")
# import os
# print(os.path.getsize('checkpoints/sam2_hiera_base_plus.pt')*0.000006, " MB")
# print(os.path.getsize('checkpoints/sam2_hiera_base_plus_dq8.pt')*0.000006, " MB")
# quantized_sam2


# image load
img = cv2.imread('/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/mv_frame_1734764824.134221.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)


cfg = {
    "file_path":"100.jpg",
    "points_per_side":36,
    "pred_iou_thresh":0.86,
    "stability_score_thresh":0.9,
    "crop_n_layers":2,
    "crop_n_points_downscale_factor":1,
    "box_nms_thresh":0.8,
    "min_mask_region_area":10
}

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)







start.record()
# start = time.time()
# mask generator load
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


        # SAM inference

        try:
            result_SAM = mask_generator_SAM.generate(img_rgb)
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


    # SAM inference

    try:
        result_SAM = mask_generator_SAM.generate(img_rgb)
    except Exception as e:
        print(e)
     
end.record()
# Waits for everything to finish running
torch.cuda.synchronize()
# done = time.time()
# inference_time_SAM = done - start

print(f"inference_time_SAM : {start.elapsed_time(end)/(10**3)} sec")
# print(f"inference_time_SAM : {inference_time_SAM:.2f} sec")


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


plt.imshow(result_image)
plt.imsave('result/gs_base_plus.jpg', result_image)


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