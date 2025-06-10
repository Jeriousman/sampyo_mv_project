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
import matplotlib.pyplot as plt
import traceback
import requests
from requests.auth import HTTPDigestAuth

import logging.handlers

from PIL import Image

import timm
import torch
import pyodbc
import numpy as np
import pandas as pd
import cv2 

from torchvision import transforms





from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from arena_api.system import system
import subprocess
import statistics
from datetime import timedelta

import pickle
sys.path.append("/home/sdt/Workspace/onvif/python-onvif-zeep/socket")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import mssql

######################################################################
#                             Save Log                               #
######################################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
        filename=f"/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/logs/socket_data_mv.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.addHandler(log_fileHandler)

######################################################################
#                              Config                                #
######################################################################
# with open('./config/server_info.json', 'r') as f:
with open('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/config/server_info.json', 'r') as f:
    cfg = json.load(f)

SEED = cfg['model']['seed']




# SERVER_HOST = cfg['server_host']
# SERVER_PORT = 7702



# seq_no_dict = {}
# TODAY = datetime.date.today()
# YESTERDAY = TODAY - timedelta(days=1)

# ######################################################################
# #                              Config                                #
# ######################################################################
# # with open('/config/db_info.json', 'r') as f:
# with open('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/config/db_info.json', 'r') as f:
#     db_info = json.load(f)

# # with open('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/config/db_info.json', 'r') as f:
# #     db_info = json.load(f)


# SAMPYO_HOST = db_info['sampyo_host']
# SAMPYO_DBNAME = db_info['sampyo_dbname']
# SAMPYO_USERNAME = db_info['sampyo_username']
# SAMPYO_PASSWORD = db_info['sampyo_password']

# SDT_HOST = db_info['sdt_host']
# SDT_DBNAME = db_info['sdt_dbname']
# SDT_USERNAME = db_info['sdt_username']
# SDT_PASSWORD = db_info['sdt_password']


######################################################################
#                             Fix seed                               #
######################################################################
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


######################################################################
#                   Check duplication in veqs_no                     #
######################################################################
def is_duplicated(vseq_no):
    global seq_no_dict

    if vseq_no in seq_no_dict:
        return True
    else:
        seq_no_dict[vseq_no] = 1
        return False


######################################################################
#            Initialize dictionary for check duplication             #
######################################################################
def initialize_vseq_no():
    global sqe_no_dict, TODAY, YESTERDAY

    TODAY = datetime.date.today()
    if str(YESTERDAY) != str(TODAY):
        YESTERDAY = TODAY
        seq_no_dict = {}
        logger.info(f'Initialize vseq_no dictionary.')



######################################################################
#                              Camera                                #
######################################################################

class MVCam:
    def __init__(self):
        try:
            self.device_infos = system.device_infos
            #self.device = system.create_device(device_infos=self.device_infos[0])[0]

            self.devices = create_devices_with_tries()
            self.device = system.select_device(self.devices)        

            #self.device.start_stream()

            ########## added by dykim ##########
            # Get device stream nodemap
            tl_stream_nodemap = self.device.tl_stream_nodemap
            # Enable stream auto negotiate packet size
            tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
            # Enable stream packet resend
            tl_stream_nodemap['StreamPacketResendEnable'].value = True
            # Get/Set nodes -----------------------------------------------------------
            #nodes = self.device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])
            nodes = self.device.nodemap.get_node(['Width', 'Height', 'PixelFormat',  'ExposureAuto', 'ExposureTime','DeviceStreamChannelPacketSize','OffsetX','OffsetY',])
            logger.info(f"self.device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
            
            nodes['OffsetX'].value = 0
            nodes['OffsetY'].value = 0
            logger.info(f"self.device.nodemap - OffsetX, OffsetY : {nodes['OffsetX'].value}, {nodes['OffsetY'].value}")
        
            # to acquisition rapid, packet size setting: origin = 1500
            stream_packet_size_max = nodes['DeviceStreamChannelPacketSize'].max
            nodes['DeviceStreamChannelPacketSize'].value = stream_packet_size_max
            logger.info(f"self.device.nodemap - DeviceStreamChannelPacketSize, packet size max : {nodes['DeviceStreamChannelPacketSize'].value}, {nodes['DeviceStreamChannelPacketSize'].max}")
        
        
            #print(f"self.device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
            # set exposure time: origin=16394.6
            nodes['ExposureTime'].value = 5000.0
            #print(f"self.device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
            # Set pixel format to Mono8
            pixel_format_name = 'BayerRG8'
            #logger.info(f'Setting Pixel Format to {pixel_format_name}')
            nodes['PixelFormat'].value = pixel_format_name
            
            # setting width, height
            print('Setting Width to 2700')
            nodes['Width'].value = 2700
            print('Setting Height to 2000')
            height = nodes['Height']
            height.value = 2000

            # demonstrages save: mono8 to png
            #self.device.start_stream(1)
            #logger.info('Started stream')
        except:
            logger.error(traceback.format_exc())
            #self.device.stop_stream()
            system.destroy_device()
            logger.error(traceback.format_exc())
            logger.error('Destroyed all created devices')
           



    def capture(self):
    # def capture(self):
        try:
            ### added by dykim ###
            self.device.start_stream(1)
            logger.info('Started stream')
            logger.info('Grabbing an image buffer')
            image_buffer = self.device.get_buffer()
            # print(f' Width X Height = ' 
            #     f'{image_buffer.width} x {image_buffer.height}')
            
            ### test method 1 ###
            # image_only_data = None
            # if image_buffer.has_chunkdata:
            #     # 8 is the number of bits in a byte
            #     bytes_pre_pixel = int(image_buffer.bits_per_pixel / 8)

            #     image_size_in_bytes = image_buffer.height * \
            #         image_buffer.width * bytes_pre_pixel

            #     image_only_data = image_buffer.data[:image_size_in_bytes]
            # else:
            #     image_only_data = image_buffer.data

            # nparray = np.asarray(image_only_data, dtype=np.uint8)
            # # Reshape array for pillow
            # nparray_reshaped = nparray.reshape((
            #     image_buffer.height,
            #     image_buffer.width))
            
            #### method 2 ####  더 일반적이라는 안내가 있지만, 예제에서 사용되지 않았으므로, 추후에 테스트 해봐야 함 
            nparray_reshaped = np.ctypeslib.as_array(
               image_buffer.pdata,(image_buffer.height, image_buffer.width))
            
            rgb_img = cv2.cvtColor(nparray_reshaped, cv2.COLOR_BayerRG2BGR)

            # img post processing 
            brightness_value = 50 # 밝기 증가 값 
            clipLimit_value = 3.0 # 대비 향상 값 
            img_brightened = cv2.add(rgb_img, np.full(rgb_img.shape, brightness_value, dtype=np.uint8))
            # 선명도 조절
            # BGR을 LAB 색 공간으로 변환
            lab = cv2.cvtColor(img_brightened, cv2.COLOR_BGR2LAB)
            # L 채널을 분리
            l, a, b = cv2.split(lab)
            # CLAHE 적용 (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=clipLimit_value, tileGridSize=(8,8))
            l_clahe = clahe.apply(l)
            # 다시 병합하여 LAB → BGR 변환
            lab_clahe = cv2.merge((l_clahe, a, b))
            enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


            #### save img ####
            print('Saving image')
            current_time_mv = datetime.datetime.now()
            logger.info(f"MV rgb img saved at : {current_time_mv}") 

            mv_img_name = f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_{current_time_mv}.jpg"
            mv_img_array = Image.fromarray(enhanced_image)
            mv_img_array.save(mv_img_name)
            print(f'Saved image path is: {mv_img_name}')

            ### original code ###
            # image_buffer = self.device.get_buffer()
            # current_time_mv = datetime.datetime.now()
            # logger.info(f"MV rgb img saved at : {current_time_mv}") 
            # ##ctypeslib is a module that makes C library run in python. ##pdata = pixel data?
            # ##(3000, 4096, 1) shape
            # np_img = np.ctypeslib.as_array(image_buffer.pdata,shape=(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))
            # ##(3000, 4096, 3) shape. Grayscale to RGB 
            # rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BayerRG2RGB)
            # # rgb_img.shape
            # #cv2.imshow('lucid_img', rgb_img)
            # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_{current_time_mv}_{vseq_no}.jpg", rgb_img)
            # # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_{current_time_mv}.jpg", rgb_img)
            
            
            self.device.requeue_buffer(image_buffer)
            print('requeue buffer')
            self.device.stop_stream()
            print('stop stream after requeue buffer')
            return mv_img_array
        
        except Exception as e:
            self.device.stop_stream()
            print(e)
            logger.info(traceback.format_exc())
        



'''
Acquisition: Rapid Acquisition
	This example demonstrates configuring device settings in order to reduce
	bandwidth and increase framerate. This includes reducing the region of
	interest, reducing bits per pixel, increasing packet size, reducing exposure
	time, and setting a large number of buffers.
'''
TAB1 = "  "
TAB2 = "    "
'''
=-=-=-=-=-=-=-=-=-
=-=- EXAMPLE -=-=-
=-=-=-=-=-=-=-=-=-
'''


def create_devices_with_tries():
	'''
	Waits for the user to connect a device before raising an
		exception if it fails
	'''
	tries = 0
	tries_max = 100
	sleep_time_secs = 1
	devices = None
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			print(
				f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				print(f'{TAB1}{sec_count + 1 } seconds passed ',
					'.' * sec_count, end='\r')
			tries += 1
		else:
			print(f'{TAB1}Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'{TAB1}No device found! Please connect a device and run '
						f'the example again.')






######################################################################
#                     Granular Analysis Model                        #
######################################################################


# class granule_sam:
#     def __init__(self):

#         self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.model_name = 'sam2.1_hiera_large'
#         self.mixed = True
#         self.checkpoint = f"./checkpoints/{self.model_name}.pt"
#         self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#         # self.model_cfg = "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/granule_analysis/sam2_hiera_l.yaml"
#         # self.checkpoint = "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/granule_analysis/checkpoints/sam2.1_hiera_large.pt"

#         self.cfg = {
#             "file_path":"100.jpg",
#             "points_per_side":40,
#             "pred_iou_thresh":0.86,
#             "stability_score_thresh":0.9,
#             "crop_n_layers":2,
#             "crop_n_points_downscale_factor":1,
#             "box_nms_thresh":0.8,
#             "min_mask_region_area":10
#         }
#         self.inference_result_path = '/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/sampyo_sam2_2.jpg'

#         sam2 = build_sam2(self.model_cfg, self.checkpoint, device = self.DEVICE, apply_postprocessing=False)
#         ## For faster inference, we use torch.compile
#         self.sam2 = torch.compile(sam2, mode='reduce-overhead')

#     def forward_granule(self, img_path):
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_w, img_h, img_c = img.shape
#         print(f'image shape: {img.shape}')
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         start.record()
#         # mask generator load
#         if self.mixed:
#             # result_path = f'result/gs_test_{model_name}_mixed.jpg'
#             with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#                 mask_generator_SAM = SAM2AutomaticMaskGenerator(
#                     model = self.sam2,
#                         points_per_side = self.cfg['points_per_side'],
#                         pred_iou_thresh=self.cfg['pred_iou_thresh'],
#                         stability_score_thresh=self.cfg['stability_score_thresh'],
#                         crop_n_layers=self.cfg['crop_n_layers'],
#                         crop_n_points_downscale_factor=self.cfg['crop_n_points_downscale_factor'],
#                         box_nms_thresh=self.cfg['box_nms_thresh'],
#                         min_mask_region_area=self.cfg['min_mask_region_area'],
#                     )
#                 # SAM inference
#                 try:
#                     result_SAM = mask_generator_SAM.generate(img)
#                 except Exception as e:
#                     print(e)
#         else:
#             # result_path = f'result/gs_test_{model_name}.jpg'
#             mask_generator_SAM = SAM2AutomaticMaskGenerator(
#                 model = self.sam2,
#                     points_per_side = self.cfg['points_per_side'],
#                     pred_iou_thresh=self.cfg['pred_iou_thresh'],
#                     stability_score_thresh=self.cfg['stability_score_thresh'],
#                     crop_n_layers=self.cfg['crop_n_layers'],
#                     crop_n_points_downscale_factor=self.cfg['crop_n_points_downscale_factor'],
#                     box_nms_thresh=self.cfg['box_nms_thresh'],
#                     min_mask_region_area=self.cfg['min_mask_region_area'],
#                 )
#             # SAM inference
#             try:
#                 result_SAM = mask_generator_SAM.generate(img)
#             except Exception as e:
#                 print(e)     
#         end.record()
#         # Waits for everything to finish running
#         torch.cuda.synchronize()
#         print(f"inference_time_SAM : {start.elapsed_time(end)/(10**3)} sec")

#         # SAM segmentation
#         # cumulated_SAM= np.zeros(result_SAM[1]["segmentation"].shape)
#         shape = result_SAM[0]['segmentation'].shape
#         result_image = np.zeros(shape)
#         #real_size = ()**2
#         count = 0
#         counts_SAM = 0
#         start = time.time()
#         for n, r in enumerate(result_SAM):
#             if r['area'] < 0.1 * img_w * img_h and r['stability_score'] > self.cfg['stability_score_thresh']:
#                 if np.amax(result_image + r['segmentation'].astype(int)) < 2:
#                 # if np.amax(result_image + r['segmentation'].astype(int)) < 10:
#                     result_image = result_image + r['segmentation'].astype(int)
#                     count += 1
#                 counts_SAM += 1

#         # np.amax(result_image)
#         # 시각화
#         plt.figure(figsize=(6,6))
#         # plt.imshow(cumulated_SAM)
#         #plt.title(f'inference_time: {inference_time_SAM:.2f}sec \n count: {counts_SAM}', fontsize=16)
#         plt.imshow(result_image)
#         # cv2.imwrite(self.inference_result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)) 
#         plt.imsave(self.inference_result_path, result_image)

#         return result_image

#         ##다연님이 보내준 겹치지 않게 화긴하는 방법 
#         # for n, r in enumerate(result):
#         #     if r['area'] < self.cfg['area_thresh'] * shape[0] * shape[1] and r['stability_score'] > self.cfg['stability_score_thresh']:
#         #         if np.amax(result_image + r['segmentation'].astype(int)) < 2:
#         #             result_image = result_image + r['segmentation'].astype(int)
#         #             count += 1
#         #             convert_mm2 = (np.sum(r['segmentation'])*(real_size**2))/10**6
#         #             sizes.append(round(convert_mm2,2))


#     def analize_granule(self, image):

#         mm13 = 0
#         mm20 = 0
#         mm25 = 0
#         mm40 = 0

#         # image = cv2.imread('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/sampyo_sam2.jpg')
        
#         # Convert the image to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Get the mask (predefined threshold)
#         mask = cv2.inRange(gray, 200, 255)

#         #Find all Objects contour's 
#         contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         granule_num_px_list = []
#         granule_analysis_dict = {}
#         for i in range(len(contours)):
#             m=max(contours[i][0])
#             print(area)
#             area = cv2.contourArea(contours[i])
#             # print(f"Area{i+1} = {area} pixels")
#             image = cv2.drawContours(image, contours, i, (0,255,255*i), -1)
#             img_text = cv2.putText(image, str(area), m, cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1, cv2.LINE_AA, False)


#             if area < 1450:
#                 mm13+=1 
#             elif area >= 1700 and area < 2100:
#                 mm20+=1
#             elif area >= 2300 and area < 2800:
#                 mm25+=1
#             elif area >= 3000:
#                 mm40+=1
#             granule_num_px_list.append(area)

#         min_value = min(granule_num_px_list) #0
#         max_value = max(granule_num_px_list) #6201
#         average_value = int(np.round(sum(granule_num_px_list)/len(granule_num_px_list))) ## average ##855
#         median_value = statistics.median(granule_num_px_list) ## 603.5
#         stdev_value = statistics.stdev(granule_num_px_list)
#         num_granule = len(contours) ##1691
#         # median_value/25 ##1mm당 pixel 갯수 
#         # 94.32*13
#         # 94.32*20
#         # 94.32*25
#         # 94.32*40
#         granule_analysis_dict = {"13mm": mm13, '20mm': mm20, '25mm': mm25, '40mm': mm40}

#         return granule_analysis_dict, min_value, max_value, average_value, median_value, num_granule, stdev_value
# g_model = granule_sam()
# /home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/mv_image_dy.png
# import statistics

# #######################################################################################
# ## SAM2 settings

# ## Nvidia GPU CUDA check
# print(torch.cuda.is_available())
# ## Set Variables
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ## Checking if device is CUDA
# #print(DEVICE)

# model_name = 'large2.1'
# ## Mixed precision (fp16 and fp32). If True, use mixed precision.
# mixed = True

# ## Pre-trained model and cofig path set
# checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# #checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
# #model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
# #checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
# #model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# ## sam2 configs. It can be changed up to your taste.
# cfg = {
#     "points_per_side":36,
#     "pred_iou_thresh":0.86,
#     "stability_score_thresh":0.9,
#     "crop_n_layers":2,
#     "crop_n_points_downscale_factor":1,
#     "box_nms_thresh":0.8,
#     "min_mask_region_area":10
# }
# #######################################################################################
# ## Initializing sam2 model
# sam2 = build_sam2(model_cfg, checkpoint, device =DEVICE, apply_postprocessing=False)
# ## For faster inference, we use torch.compile
# sam2 = torch.compile(sam2, mode='reduce-overhead')



if __name__ == "__main__":
    # Fix Seed
    seed_everything(SEED)
    

    # Load Camera
    mv_cam = MVCam()
    logger.info('MV cam initialized')
    print("MV cam initialized")
    rgb_img=mv_cam.capture() ##이것이 이미지를 캡쳐하는 함수이다 
    print('done')
    system.destroy_device() ##destroy mv cam
    # logger.info(f"MV camera init: {mv_cam}")

    # model = granule_sam()
    # logger.info('SAM model initialized')
    # print('SAM model initialized')
    # ##/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-01-14 14:28:46.443379.jpg



    # # MSSQL DB Connector
    # sampyo_msdb = mssql.DB(host=SAMPYO_HOST,
    #                        dbname=SAMPYO_DBNAME,
    #                        username=SAMPYO_USERNAME,
    #                        password=SAMPYO_PASSWORD)

    # sdt_msdb = mssql.DB(host=SDT_HOST,
    #                     dbname=SDT_DBNAME,
    #                     username=SDT_USERNAME,
    #                     password=SDT_PASSWORD)
    # logger.info('sdt_msdb initinated')




    # # Open Socket Server
    # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    # server_socket.bind((SERVER_HOST, SERVER_PORT))
    # server_socket.listen()
    # logger.info(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")


                                                        
    # while True:
    #     try:
    #         client_socket, client_address = server_socket.accept()
    #         logger.info(f"[*] Accepted connection from {client_address[0]}:{client_address[1]}")

    #     # Connect to device and start pipeline
    #     # with depth_camera.device:
    #     #     depth_camera.device.startPipeline(depth_camera.pipeline)
    #         while True:
    #             initialize_vseq_no()  # Initialize dictionary for data duplication check.
            
    #             data = client_socket.recv(1024)  # type: bytes
    #             if not data:
    #                 break

    #             print(data)
    #             str_data = data.decode('cp949')  # type: str
                
    #             logger.info(f"[*] Received data: {str_data}")
    #             vseq_no = str_data.split('|')[1]

    #             vin_date = datetime.datetime.now()
    #             if not is_duplicated(vseq_no):

    #                 now_datetime = datetime.datetime.now()
    #                 logger.info(f"sensor trigger received at {now_datetime}")
    #                 now_unix = int(now_datetime.timestamp())
    #                 now_strf = now_datetime.strftime("%Y%m%d-%H%M%S")
    #                 logger.info(f"Trigger type: {str_data.split('|')[-1][0]}")

    #                 if len(str_data[1:-1].split('|')) == 7:
    #                     vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url = str_data[1:-1].split('|')

    #                 elif len(str_data[1:-1].split('|')) == 8:
    #                     vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url, _ = str_data[1:-1].split('|')
                    
    #                 logger.info(f"container vseq_no: {vseq_no}")
    #                 logger.info(f"container vplant_code: {vplant_code}")
    #                 logger.info(f"container car_num: {car_num}")
    #                 logger.info(f"container item_code: {item_code}")
    #                 logger.info(f"container type_code: {type_code}")
    #                 logger.info(f"container from_code: {from_code}")

    #                 #sdt_msdb.categories['item']['31013']        
    #                 if item_code not in ["31013", "31019", "31025", "31067", "32001"]:
    #                     logger.info(f"filtered container item_code: {item_code}")
    #                     continue

    #                 elif type_code not in ["S1", "B1", "S2"]:
    #                     logger.info(f"filtered container type_code: {type_code}")
    #                     continue

    #                 # sdt_msdb.categories['from']['31013']
    #                 elif from_code not in ["0493870000", "0669560000", "0712360000",
    #                             "0736040000", "0791340000", "0892200000", "0935600000",
    #                             "1326700000", "1500680000", "1512120000", "1512730000",
    #                             "1920700000", "2298820000", "RC10017"]:
    #                     logger.info(f"filtered container from_code: {from_code}")
    #                     continue
                    

    #                 #############This is MV cam capturing image#########################
    #                 rgb_img=mv_cam.capture(vseq_no, item_code) ##이것이 이미지를 캡쳐하는 함수이다 
    #                 # # logger.info(rgb_img) 

    #                 sam_result_img = model.forward_granule(rgb_img)
    #                 granule_data = model.analize_granule(sam_result_img)


    #                 # update inference result to DB
    #                 # sampyo_msdb.update(str_data, CATEGORIES[pred])
    #                 # sdt_msdb.insert(str_data, vin_date, CATEGORIES[pred], CATEGORIES_ITEM[pred_item] ,image_path, CATEGORIES_QC[pred_qc])
    #                 # sdt_msdb.upsert_mv(str_data, granular_info)
    #                 # save inference info






    #         logger.info(f"[*] Disconnected connection from {client_address[0]}:{client_address[1]}")
    #         client_socket.close()
            
    #     except KeyboardInterrupt:
    #         server_socket.close()
    #         cv2.destroyAllWindows()
    #         system.destroy_device() ##destroy mv cam
            

    #         exit()

    #     except ConnectionResetError:
    #         logger.info(f"[*] Client Disconnected.")

    #     except Exception as e:
    #         logger.error(traceback.format_exc())
    #         server_socket.close()
    #         exit()

# system.destroy_device() ##destroy mv cam



















#     ct = 0
#     with depth_camera.device:
#         depth_camera.device.startPipeline(depth_camera.pipeline)
                                                            
#         while True:
#             ct+=1
#             if ct > 100:
#                 logger.info("break now")
#                 break
                


# #############DEPTH CAMERA START########################
#             depth_camera.messages = depth_camera.device.getOutputQueue("rgbd", maxSize=1).tryGetAll()    
#             # while True:
#             start = time.time() 

#             if not depth_camera.messages:
#                 continue
#             # currentDateAndTime = datetime.datetime.now()
#             logger.info(f"depth_camera.messages: {depth_camera.messages}")
#             for message_group in depth_camera.messages:
#                 logger.info(f"message_group: {message_group}")
#                 for name, frame in message_group:
#                     logger.info(f"{name}: {frame.getSequenceNum()}")
#                     logger.info(f"name: {name}")
#                     logger.info(f"frame: {frame}")

#                     if name == 'depth':
#                         # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
#                         depth_frame = frame.getCvFrame()##.astype(np.uint16)
#                         current_time_depth_depth = datetime.datetime.now()
#                         logger.info(f"Depth depth img saved at : {current_time_depth_depth}") 
#                         # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}_{vseq_no}.jpg", depth_frame)
#                         # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}.jpg", depth_frame)
#                         depth_time = datetime.datetime.now()
#                         # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{depth_time}_{vseq_no}.jpg", depth_frame)
#                         # # print('depth iamge collected')
#                         with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_{depth_time}_{vseq_no}.pickle", 'wb') as f:
#                             pickle.dump(depth_frame, f)
#                     ###########################################################################################
#                     ## uncomment this for depth+RGB blending
#                         # depth_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
#                         # depth_frame = cv2.equalizeHist(depth_frame)
#                         # depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)
#                     ###########################################################################################
#                         # print('depth: ', depth_frame)
#                         # print('depth: ', depth_frame.shape)

#                     # elif name == 'disparity':
#                     #     disparity_frame = frame.getCvFrame()
#                     #     cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparity_frame_{currentDateAndTime}.jpg", disparity_frame)

#                     elif name == 'rgb':
#                         rgb_frame = frame.getCvFrame()##.astype(np.uint16)
#                         current_time_depth_rgb = datetime.datetime.now()
#                         logger.info(f"Depth rgb img saved at : {current_time_depth_rgb}") 
#                         rgb_frame = cv2.resize(rgb_frame, (depth_camera.h, depth_camera.w), interpolation=cv2.INTER_NEAREST)
#                         # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}_{vseq_no}.jpg", rgb_frame)
#                         # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}.jpg", rgb_frame)
#                         cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{datetime.datetime.now()}_{vseq_no}.jpg", rgb_frame)
#                         # print('RGB: ', rgb_frame)
#                         # print('RGB iamge collected')
#                         # print('RGB: ', rgb_frame.shape)
            
#                     end = time.time()
#                     print(end - start)