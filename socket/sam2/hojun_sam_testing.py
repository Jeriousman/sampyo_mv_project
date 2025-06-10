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
# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from maskdino import add_maskdino_config
from predictor import VisualizationDemo
              
from detectron2.data.datasets import register_coco_instances

register_coco_instances("sampyo_depth_train", {}, "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/datasets/instances_train2017.json", "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/datasets/train2017")
register_coco_instances("sampyo_depth_val", {}, "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/datasets/instances_val2017.json", "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/datasets/val2017")

metadata_train = MetadataCatalog.get("sampyo_depth_train")
metadata_val = MetadataCatalog.get("sampyo_depth_val")
metadata_train.set(thing_classes=['container',])
metadata_val.set(thing_classes=['container',])


def setup_cfg():
    # load config from file and command-line arguments
    config_file = "/home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"
    opts = ['MODEL.WEIGHTS', '/home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/demo/model_final_sampyo_250204.pth']
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg

mp.set_start_method("spawn", force=True)
cfg = setup_cfg()
predictor_vis = VisualizationDemo(cfg)

######################################################################
#                             Save Log                               #
######################################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
        filename=f"/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/logs/socket_data.log",
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
MODEL_NAME = cfg['model']['name']
NUM_CLASSES = cfg['model']['num_classes']
NUM_CLASSES_QC = cfg['model']['num_classes_qc']
NUM_CLASSES_ITEM = cfg['model']['num_classes_item']
DEVICE = cfg['model']['device']
MODEL_CKPT = cfg['model']['ckpt_path']
MODEL_CKPT_QC = cfg['model']['ckpt_path_qc']
MODEL_CKPT_ITEM = cfg['model']['ckpt_path_item']
CATEGORIES = {0: '모래',
              1: '자갈',
              2: '덮개',
              3: '빈차',
              4: '레미콘',
              5: '차량없음'}

CATEGORIES_QC = {0: '부족',
                 1: '정상'}

CATEGORIES_ITEM = {0: '석산',
                   1: '석산',
                   2: '발파석',
                   3: '발파석'}

IMAGE_BUCKET = cfg['image_bucket']

SERVER_HOST = cfg['server_host']
SERVER_PORT = cfg['server_port']

CAM_IP = cfg['cam_ip']
CAM_PORT = cfg['cam_port']
CAM_ID = cfg['cam_id']
CAM_PW = cfg['cam_pw']


seq_no_dict = {}
TODAY = datetime.date.today()
YESTERDAY = TODAY - timedelta(days=1)

######################################################################
#                              Config                                #
######################################################################
# with open('/config/db_info.json', 'r') as f:
with open('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/config/db_info.json', 'r') as f:
    db_info = json.load(f)

# with open('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/config/db_info.json', 'r') as f:
#     db_info = json.load(f)


SAMPYO_HOST = db_info['sampyo_host']
SAMPYO_DBNAME = db_info['sampyo_dbname']
SAMPYO_USERNAME = db_info['sampyo_username']
SAMPYO_PASSWORD = db_info['sampyo_password']

SDT_HOST = db_info['sdt_host']
SDT_DBNAME = db_info['sdt_dbname']
SDT_USERNAME = db_info['sdt_username']
SDT_PASSWORD = db_info['sdt_password']


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
#            Initialize dictionary for check duplication             #
######################################################################
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        # load image
        image = Image.open(image_path)
        
        # resize image
        width, height = image.size
        resized_image = image.resize((width // 10, height // 10))

        io_buffer = io.BytesIO()
        resized_image.save(io_buffer, format='JPEG')
        
        # encode image
        encoded_image = base64.b64encode(io_buffer.getvalue())

    return encoded_image.decode()


######################################################################
#                            DB Update                               #
######################################################################
# def db_update(vplant_code, vseq_no, item):
#     connection_string = f'DRIVER={{{DB_DRIVER}}};SERVER={DB_HOST};DATABASE={DB_DBNAME};UID={DB_USERNAME};PWD={DB_PASSWORD}'
# 
#     connection = pyodbc.connect(connection_string)
#     cursor = connection.cursor()
# 
#     query = f"UPDATE E_INOUTLOG SET SDT_ITEM='{item}' WHERE VPLANT_CODE='{vplant_code}' AND VSEQ_NO={vseq_no}"
# 
#     cursor.execute(query)
#     connection.commit()
# 
#     logger.info(f"DB UPDATE. SDT_ITEM='{item}'")
# 
#     cursor.close()
#     connection.close()


######################################################################
#                              Camera                                #
######################################################################
class ONVIFCam:
    def __init__(self):
        self.cam = ONVIFCamera(CAM_IP, CAM_PORT, CAM_ID, CAM_PW)
        self.media_service = self.cam.create_media_service()
        self.media_profile = self.media_service.GetProfiles()[0]

    def get_snapshot_and_save(self, now_strf):
        logger.info(f'ptz get snapshot start!')
        snapshot_uri = self.media_service.GetSnapshotUri({'ProfileToken': self.media_profile.token}).Uri
        response = requests.get(snapshot_uri, auth=HTTPDigestAuth(CAM_ID, CAM_PW))

        if response.status_code == 200:
            file_path = f'{now_strf}.jpg'
            save_path = os.path.join(IMAGE_BUCKET, str(TODAY))

            if not os.path.exists(save_path):
                os.makedirs(save_path)
             
            image_path = os.path.join(save_path, file_path)
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info(f'ptz get snapshot end!')
        
        else:
            logger.info("Failed to fetch snapshot")

        return os.path.abspath(image_path)



#cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLOMAP_JET)
#cvColorMap[0] = [0,0,0]

class depth_cam:
    def __init__(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()
        self.device = dai.Device()
        self.fps = 30

        # The disparity is computed at this resolution, then upscaled to RGB resolution
        #monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        self.monoResolution = dai.MonoCameraProperties.SensorResolution.THE_800_P
        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.left = self.pipeline.create(dai.node.MonoCamera)
        self.right = self.pipeline.create(dai.node.MonoCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)

        try:
            calibData = self.device.readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lensPosition:
                self.camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise

        #Properties
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        # self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP) # 4056x3040
        self.camRgb.setFps(self.fps)

        self.left.setResolution(self.monoResolution)
        self.left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        self.left.setFps(self.fps)
        self.right.setResolution(self.monoResolution)
        self.right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        self.right.setFps(self.fps)

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # LR-check is required for depth alignment
        self.stereo.setLeftRightCheck(True)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        # self.h, self.w = 1248*2, 936*2
        self.h, self.w = 1280, 720
        # self.h, self.w = 4048, 3040
        # # 4056x3040
        self.stereo.setOutputSize(self.h, self.w)
        #stereo.setOutputSize(1280, 720)
        # stereo.setOutputSize(4056, 3040)

        ##################Use this when using Sync##################
        self.sync = self.pipeline.create(dai.node.Sync)
        self.sync.setSyncThreshold(timedelta(milliseconds=50))
        self.sync.inputs["rgb"].setQueueSize(1)
        self.sync.inputs["depth"].setQueueSize(1)
        self.sync.inputs["disparity"].setQueueSize(1)
        
        # Linking
        self.camRgb.isp.link(self.sync.inputs["rgb"])
        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)
        self.stereo.depth.link(self.sync.inputs["depth"])
        self.stereo.disparity.link(self.sync.inputs["disparity"])

        self.sync_out = self.pipeline.createXLinkOut()
        self.sync_out.setStreamName("rgbd")
        self.sync.out.link(self.sync_out.input)
        self.sync_out.input.setQueueSize(1)
        ##############################################################
        ##################Use this when NOT using Sync##################
        # # # Linking
        # # self.xoutRgb = self.pipeline.createXLinkOut()
        # self.xoutRgb = self.pipeline.create(dai.node.XLinkOut) ##바로위에거랑 같은뜻이다. 
        # self.xoutdepth = self.pipeline.createXLinkOut()
        # self.xoutdisparity = self.pipeline.createXLinkOut()
        # # self.xoutdepth.setStreamName("depth")
        # # self.xoutdisparity.setStreamName("disparity")

        # self.left.out.link(self.stereo.left)
        # self.right.out.link(self.stereo.right)
        # self.stereo.disparity.link(self.camRgb.isp)
        # self.camRgb.isp.link(self.xoutdisparity.input)
        # self.stereo.depth.link(self.xoutdepth.input)
    
        ##############################################################

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
           



    def capture(self, vseq_no):
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

            mv_img_name = f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_{current_time_mv}_{vseq_no}.jpg"
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
#                              Model                                 #
######################################################################
class Model:
    def __init__(self, ckpt_path, num_classes):
        self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes).to(DEVICE)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        
        self.transform = transforms.Compose([transforms.Resize((384, 384)),
                                             transforms.ToTensor()])
        self.inference('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/dummy.jpg')

    def inference(self, image_path):
        image = Image.open(image_path)
        t_image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            self.model.eval()

            inputs = t_image.to(DEVICE)
            outputs = self.model(inputs)

            preds = torch.argmax(outputs, dim=-1)

        return preds.detach().cpu().numpy()[0]

###MV 입도분석 모델 loading 
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
        self.inference_result_path = '/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/sam_img'

        sam2 = build_sam2(self.model_cfg, self.checkpoint, device = self.DEVICE, apply_postprocessing=False)
        ## For faster inference, we use torch.compile
        self.sam2 = torch.compile(sam2, mode='reduce-overhead')

    def forward_granule(self, img):
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        logger.info(f'granule image shape: {img.shape}')
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
        plt.imsave(f"{self.inference_result_path}/{datetime.datetime.now()}_sampyo_sam.jpg", result_image)
        logger.info(f"SAM result_image final output shape: {result_image.shape}")
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


    def analyze_granule(self, image):
        try:

            mm13 = 0
            mm20 = 0
            mm25 = 0
            mm40 = 0

            # image = cv2.imread('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/sampyo_sam2.jpg')
            
            # Convert the image to grayscale
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get the mask (predefined threshold)
            mask = cv2.inRange(image, 1, 255)

            #Find all Objects contour's 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            logger.info(f"contours: {contours}")
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

            

            # min_value = min(granule_num_px_list) #0
            # max_value = max(granule_num_px_list) #6201
            # average_value = int(np.round(sum(granule_num_px_list)/len(granule_num_px_list))) ## average ##855
            # median_value = statistics.median(granule_num_px_list) ## 603.5
            # stdev_value = statistics.stdev(granule_num_px_list)
            # num_granule = len(contours) ##1691
            # median_value/25 ##1mm당 pixel 갯수 
            # 94.32*13
            # 94.32*20
            # 94.32*25
            # 94.32*40
            granule_analysis_dict = {"13mm": mm13, '20mm': mm20, '25mm': mm25, '40mm': mm40}
            logger.info('granule analysis done')

            return granule_analysis_dict
            #return granule_analysis_dict, min_value, max_value, average_value, median_value, num_granule, stdev_value
        except:
            logger.error(traceback.format_exc())
            return ['None']


def main_ptz(q, model, model_qc, model_item, sampyo_msdb, sdt_msdb):
    try:
        packets = q.get()
        logger.info("PTZ queue got data")
        str_data = packets[0]
        logger.info(f"PTZ str_data: {str_data}")
        image_path = packets[1]
        logger.info(f"PTZ image_path: {image_path}")
        vin_date = packets[2]

        vseq_no = str_data.split('|')[1]
        

        # vin_date = datetime.datetime.now()
        if not is_duplicated(vseq_no):

            now_datetime = datetime.datetime.now()
            logger.info(f"sensor trigger received at {now_datetime}")
            now_unix = int(now_datetime.timestamp())
            now_strf = now_datetime.strftime("%Y%m%d-%H%M%S")
            logger.info(f"Trigger type: {str_data.split('|')[-1][0]}")

            
            # image_path = onvif_cam.get_snapshot_and_save(now_strf)
            pred = model.inference(image_path)
            pred_qc = model_qc.inference(image_path)
            pred_item = model_item.inference(image_path)
            logger.info(f"PTZ VIT inferenced item: {pred_item}")
            # update inference result to DB
            sampyo_msdb.update(str_data, CATEGORIES[pred])
            # mydb.insert(str_data, vin_date, CATEGORIfES[pred], image_path)
            #sdt_msdb.insert(str_data, vin_date, CATEGORIES[pred], CATEGORIES_ITEM[pred_item] ,image_path, CATEGORIES_QC[pred_qc])
            sdt_msdb.upsert_ptz(str_data, vin_date, CATEGORIES[pred], CATEGORIES_ITEM[pred_item] ,image_path)
            logger.info(f"sdt_msdb PTZ data DB saved")

            # save inference info
            current_inference_info = pd.DataFrame([{"timestamp": now_unix,
                                                    "image_path": image_path,
                                                    "classification_result_type": CATEGORIES[pred],
                                                    "classification_result_quantity": CATEGORIES_QC[pred_qc],
                                                    "classification_result_item": CATEGORIES_ITEM[pred_item]}])

            save_folder = f'./result/{TODAY}/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            if not os.path.exists(f'./result/{TODAY}/{now_strf}.csv'):
                current_inference_info.to_csv(f'./result/{TODAY}/{now_strf}.csv', index=False)
    except:
        logger.error(traceback.format_exc())

def main_mv(q, granule_model, sdt_msdb):
    try:
        packets = q.get()
        logger.info(f"MV queue got data")
        str_data = packets[0]
        logger.info(f"MV str_data: {str_data}")
        img = packets[1]
        logger.info(f"MV img: {img}")
        logger.info(f"MV img type: {type(img)}") ##PIL Image
        sam_result_img = granule_model.forward_granule(img)
        logger.info(f"sam_result_img shape: {sam_result_img.shape}")
        granule_data = granule_model.analyze_granule(sam_result_img)
        logger.info(f'=== granule_data ===: {granule_data}')
        sdt_msdb.upsert_mv(str_data, 'None')
        #sdt_msdb.upsert_mv(str_data, str(granule_data)) 
        # return img
    except:
        logger.error(traceback.format_exc())


    # vseq_no = str_data.split('|')[1]

    # vin_date = datetime.datetime.now()
    # if not is_duplicated(vseq_no):

    #     now_datetime = datetime.datetime.now()
    #     logger.info(f"sensor trigger received at {now_datetime}")
    #     now_unix = int(now_datetime.timestamp())
    #     now_strf = now_datetime.strftime("%Y%m%d-%H%M%S")
    #     logger.info(f"Trigger type: {str_data.split('|')[-1][0]}")
    #     rgb_img=mv_cam.capture() ##이것이 이미지를 캡쳐하는 함수이다
    #     return rgb_img



def main_depth(q, sdt_msdb):
    try:
        packets = q.get()
        logger.info(f"Depth queue got data")
        # vseq_no = str_data.split('|')[1]

        # vin_date = datetime.datetime.now()
        # if not is_duplicated(vseq_no):

        #     now_datetime = datetime.datetime.now()
        #     logger.info(f"sensor trigger received at {now_datetime}")
        #     now_unix = int(now_datetime.timestamp())
        #     now_strf = now_datetime.strftime("%Y%m%d-%H%M%S")
        #     logger.info(f"Trigger type: {str_data.split('|')[-1][0]}")
        str_data = packets[0]
        logger.info(f"Depth str_data: {str_data}")
        depth_frame = packets[1]
        # logger.info(f"depth_frame: {depth_frame}")
        logger.info(f"depth_frame shape: {depth_frame.shape}")
        rgb_frame = packets[2]
        # logger.info(f"rgb_frame: {rgb_frame}")
        logger.info(f"rgb_frame shape: {rgb_frame.shape}")

        ### MaskDINO inference start ###
        while True:
            try:
                logger.info('MaskDINO inference start!')
                start = time.time()
                predictions, visualized_output = predictor_vis.run_on_image(rgb_frame)
                logger.info(f'MaskDINO inference end, inference time: {time.time() - start}')
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("CUDA OOM error! Waiting 20 seconds before retrying...")
                    time.sleep(20)
                    continue  # 다시 시도
                else:
                    raise  # OOM 이외의 에러는 그대로 발생시키기

        now_time = datetime.datetime.now()
        depth_strf = now_time.strftime("%Y%m%d-%H%M%S")
        
        # combined_mask는 instance가 2개일 때만 필요.
        if len(predictions['instances']) == 2:
            combined_mask = torch.any(torch.stack([predictions['instances'][0].pred_masks,predictions['instances'][1].pred_masks]), dim=0)
            combined_mask_np = combined_mask.numpy()
            mask_ck = True
        elif len(predictions['instances']) == 1:
            combined_mask = predictions['instances'][0].pred_masks
            combined_mask_np = combined_mask.numpy()

            mask_ck = True
        elif len(predictions['instances']) == 0: #  객체 검출이 안된 경우 예외처리를 위한 flag
            mask_ck = False
        
        # 결과 확인용 코드 (추후 사용 안해도 됨) 
        if mask_ck:
            # Mask 부분만 추출하여 저장 
            mask_np = combined_mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255 # 마스크 이미지 저장을 위한 (1, H, W) → (H, W) 변환
            cv2.imwrite(f'/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_mask/combined_mask_{depth_strf}.jpg', mask_np)
            print('Save Depth Mask')
            # MaskDINO 결과 이미지 저장 
            visualized_output.save(f'/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_mask/masking_result_{depth_strf}.jpg')
            print('Save Depth MaskDINO result img')

        else:
            logger.error("Failed to segment Depth image")

        ######## MaskDino end ###########

        interested_area = np.where((depth_frame >= 6500) & (depth_frame <= 7600), 1, 0)
        #logger.info(f'Depth interested_area: {interested_area}, shape:{interested_area.shape}')
        #logger.info(f'Depth type: interested_area = {type(interested_area)}, combined_mask_np type, shape = {type(combined_mask_np), combined_mask_np.shape}, depth_frame = {type(depth_frame)}')

        # quantity = interested_area * maskdino * (9894 - dp) ## 9894 is depth to the ground
        quantity = interested_area * combined_mask_np * (9894 - depth_frame) ## 9894 is depth to the ground
        #logger.info(f'Depth quantity: {quantity}')
        avg_quantity = quantity.mean()
        # avg_quantity = str(quantity//(interested_area.shape[0] * interested_area.shape[1]))
        logger.info(f'Depth avg_quantity: {avg_quantity}')
        sdt_msdb.upsert_depth(str_data, avg_quantity)

    except:
            logger.error(traceback.format_exc())
    
    # for message_group in depth_messages:
    #     #logger.info(f"message_group: {message_group}")
    #     for name, frame in message_group:
    #         #logger.info(f"{name}: {frame.getSequenceNum()}")
    #         #logger.info(f"name: {name}")
    #         #logger.info(f"frame: {frame}")

    #         if name == 'depth':
    #             # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
    #             depth_frame = frame.getCvFrame()##.astype(np.uint16)
                    
    #             # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}_{vseq_no}.jpg", depth_frame)
    #             depth_time = datetime.datetime.now()

    #             # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{depth_time}_{vseq_no}.jpg", depth_frame)
    #             # with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depth_frame_array_{depth_time}_{vseq_no}.pickle", 'wb') as f:

    #             # with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depth_frame_array_{depth_time}_{item_code}_{type_code}_{from_code}.pickle", 'wb') as f:
    #             with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_{depth_time}.pickle", 'wb') as f:
    #                 pickle.dump(depth_frame, f)
    #         ###########################################################################################
    #         ## uncomment this for depth+RGB blending
    #             # depth_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    #             # depth_frame = cv2.equalizeHist(depth_frame)
    #             # depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)
    #         ###########################################################################################
    #             # print('depth: ', depth_frame)
    #             # print('depth: ', depth_frame.shape)

    #         elif name == 'disparity':
    #             disparity_time = datetime.datetime.now()
    #             disparity_frame = frame.getCvFrame()

    #             # maxDisp = depth_camera.stereo.initialConfig.getMaxDisparity()
    #             # disp = (disparity_frame * (255.0/maxDisp)).astype(np.uint8)
    #             # disp = cv2.applyColorMap(disp, cvColorMap)
    #             cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparityFrame_{disparity_time}.jpg", disparity_frame)
    #             # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparityFrame_{disparity_time}_{item_code}_{type_code}_{from_code}.jpg", disparity_frame)
                
    #         elif name == 'rgb':
    #             rgb_frame = frame.getCvFrame()##.astype(np.uint16)
    #             current_time_depth_rgb = datetime.datetime.now()
    #             rgb_frame = cv2.resize(rgb_frame, (depth_camera.h, depth_camera.w), interpolation=cv2.INTER_NEAREST)
    #             # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}_{vseq_no}.jpg", rgb_frame)
    #             # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{datetime.datetime.now()}_{vseq_no}.jpg", rgb_frame)
    #             # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgbFrame_{current_time_depth_rgb}_{item_code}_{type_code}_{from_code}.jpg", rgb_frame)                                    
    #             cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgbFrame_{current_time_depth_rgb}.jpg", rgb_frame)                                    




if __name__ == "__main__":
    # Fix Seed
    seed_everything(SEED)

    # Load Camera
    onvif_cam = ONVIFCam()
    logger.info(f"onvif_cam initialized: {onvif_cam}")
    depth_camera = depth_cam()
    print("Depth cam initialized")
    logger.info(f"Depth camera init: {depth_camera}")
    mv_cam = MVCam()
    print("MV cam initialized")
    logger.info(f"MV camera init: {mv_cam}")
    granule_model = granule_sam()
    print("Granule_model initialized")
    logger.info(f"Granule_model init: {granule_model}")

    # MSSQL DB Connector
    sampyo_msdb = mssql.DB(host=SAMPYO_HOST,
                           dbname=SAMPYO_DBNAME,
                           username=SAMPYO_USERNAME,
                           password=SAMPYO_PASSWORD)

    sdt_msdb = mssql.DB(host=SDT_HOST,
                        dbname=SDT_DBNAME,
                        username=SDT_USERNAME,
                        password=SDT_PASSWORD)
    logger.info('sdt_msdb initinated')

    # Load Model
    model = Model(MODEL_CKPT,NUM_CLASSES)
    logger.info('SDT model loaded')
    model_qc = Model(MODEL_CKPT_QC, NUM_CLASSES_QC)
    model_item = Model(MODEL_CKPT_ITEM, NUM_CLASSES_ITEM)

    # Open Socket Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen()
    logger.info(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")

    queue_mv = Queue()
    queue_ptz = Queue()
    queue_depth = Queue()

    with depth_camera.device:
        depth_camera.device.startPipeline(depth_camera.pipeline)
                                                            
        while True:
            try:
                client_socket, client_address = server_socket.accept()
                logger.info(f"[*] Accepted connection from {client_address[0]}:{client_address[1]}")

            # Connect to device and start pipeline
            # with depth_camera.device:
            #     depth_camera.device.startPipeline(depth_camera.pipeline)
                while True:
                    initialize_vseq_no()  # Initialize dictionary for data duplication check.
                    logger.info('Waitting for client')
                
                    data = client_socket.recv(1024)  # type: bytes
                    if not data:
                        break

                    print(data)
                    str_data = data.decode('cp949')  # type: str
                    
                    logger.info(f"[*] Received data: {str_data}")
                    vseq_no = str_data.split('|')[1]
                    send_ack = f"(stx){str_data.split('|')[0][-3:]}|{str_data.split('|')[1]}(etx)".encode()
                    client_socket.sendall(send_ack)
                    logger.info(f"Send to client : {send_ack}")


                    vin_date = datetime.datetime.now()
                    if not is_duplicated(vseq_no):

                        now_datetime = datetime.datetime.now()
                        logger.info(f"sensor trigger received at {now_datetime}")
                        now_unix = int(now_datetime.timestamp())
                        now_strf = now_datetime.strftime("%Y%m%d-%H%M%S")
                        logger.info(f"Trigger type: {str_data.split('|')[-1][0]}")

                        if len(str_data[1:-1].split('|')) == 7:
                            vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url = str_data[1:-1].split('|')

                        elif len(str_data[1:-1].split('|')) == 8:
                            vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url, _ = str_data[1:-1].split('|')
                        
                        logger.info(f"container vseq_no: {vseq_no}")
                        logger.info(f"container vplant_code: {vplant_code}")
                        logger.info(f"container car_num: {car_num}")
                        logger.info(f"container item_code: {item_code}")
                        logger.info(f"container type_code: {type_code}")
                        logger.info(f"container from_code: {from_code}")

                        #sdt_msdb.categories['item']['31013']        
                        if item_code not in ["31013", "31019", "31025", "31067", "32001"]:
                            logger.info(f"filtered container item_code: {item_code}")
                            continue

                        elif type_code not in ["S1", "B1", "S2"]:
                            logger.info(f"filtered container type_code: {type_code}")
                            continue

                        # sdt_msdb.categories['from']['31013']
                        elif from_code not in ["0493870000", "0669560000", "0712360000",
                                    "0736040000", "0791340000", "0892200000", "0935600000",
                                    "1326700000", "1500680000", "1512120000", "1512730000",
                                    "1920700000", "2298820000", "RC10017"]:
                            logger.info(f"filtered container from_code: {from_code}")
                            continue
                        

                        # #############This is MV cam capturing image#########################
                        rgb_img=mv_cam.capture(vseq_no) ##이것이 MV camera 이미지를 캡쳐하는 함수이다 
                        # #############This is PTZ cam capturing image#########################
                        image_path = onvif_cam.get_snapshot_and_save(now_strf) ##PTZ catemra image taking

                        #############DEPTH CAMERA START########################
                        # depth_camera.messages = depth_camera.device.getOutputQueue("rgbd", maxSize=1, blocking=True).tryGetAll()    
                        depth_camera.messages = depth_camera.device.getOutputQueue("rgbd", maxSize=1, blocking=False).getAll()
                        # logger.info(f"Depth got depth_camera.message: {depth_camera.message}")
                        for message_group in depth_camera.messages:
                            #logger.info(f"message_group: {message_group}")
                            for name, frame in message_group:
                                #logger.info(f"{name}: {frame.getSequenceNum()}")
                                #logger.info(f"name: {name}")
                                #logger.info(f"frame: {frame}")

                                if name == 'depth':
                                    # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                                    depth_frame = frame.getCvFrame()##.astype(np.uint16)
                                     
                                    # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}_{vseq_no}.jpg", depth_frame)
                                    depth_time = datetime.datetime.now()
                                    logger.info(f"Depth depth img saved at : {depth_time}")
                                    # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{depth_time}_{vseq_no}.jpg", depth_frame)
                                    # with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depth_frame_array_{depth_time}_{vseq_no}.pickle", 'wb') as f:
                                    logger.info(f"Depth depth array saved at : {depth_time}")
                                    with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_{depth_time}_{item_code}_{type_code}_{from_code}.pickle", 'wb') as f:
                                        pickle.dump(depth_frame, f)
                                ###########################################################################################
                                ## uncomment this for depth+RGB blending
                                    # depth_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                                    # depth_frame = cv2.equalizeHist(depth_frame)
                                    # depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)
                                ###########################################################################################
                                    # print('depth: ', depth_frame)
                                    # print('depth: ', depth_frame.shape)

                                elif name == 'disparity':
                                    disparity_time = datetime.datetime.now()
                                    disparity_frame = frame.getCvFrame()

                                    # maxDisp = depth_camera.stereo.initialConfig.getMaxDisparity()
                                    # disp = (disparity_frame * (255.0/maxDisp)).astype(np.uint8)
                                    # disp = cv2.applyColorMap(disp, cvColorMap)
                                    
                                    cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparityFrame_{disparity_time}_{item_code}_{type_code}_{from_code}.jpg", disparity_frame)
                                    
                                    

                                elif name == 'rgb':
                                    rgb_frame = frame.getCvFrame()##.astype(np.uint16)
                                    current_time_depth_rgb = datetime.datetime.now()
                                    logger.info(f"Depth rgb img saved at : {current_time_depth_rgb}") 
                                    rgb_frame = cv2.resize(rgb_frame, (depth_camera.h, depth_camera.w), interpolation=cv2.INTER_NEAREST)
                                    # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}_{vseq_no}.jpg", rgb_frame)
                                    # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{datetime.datetime.now()}_{vseq_no}.jpg", rgb_frame)
                                    cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgbFrame_{current_time_depth_rgb}_{item_code}_{type_code}_{from_code}.jpg", rgb_frame)                                    

                        
                        
                        # image_path = onvif_cam.get_snapshot_and_save(now_strf) ##PTZ catemra image taking
                        #=======================================

                     
                        qdata_depth = [str_data, depth_frame, rgb_frame]
                        qdata_mv = [str_data, rgb_img]
                        qdata_ptz = [str_data, image_path, vin_date]

                        queue_ptz.put(qdata_ptz)    
                        queue_mv.put(qdata_mv)
                        queue_depth.put(qdata_depth)


                        sampyo_processes = [
                            Process(target=main_ptz, args=(queue_ptz, model, model_qc, model_item, sampyo_msdb, sdt_msdb)),
                            Process(target=main_mv, args=(queue_mv, granule_model, sdt_msdb)),                            
                            Process(target=main_depth, args=(queue_depth, sdt_msdb))
                        ]

                        for process in sampyo_processes:
                            process.start()
                            # sensor.join()



                        # pred = model.inference(image_path)
                        # pred_qc = model_qc.inference(image_path)
                        # pred_item = model_item.inference(image_path)
                        # # update inference result to DB
                        # sampyo_msdb.update(str_data, CATEGORIES[pred])
                        # # mydb.insert(str_data, vin_date, CATEGORIES[pred], image_path)
                        # sdt_msdb.insert(str_data, vin_date, CATEGORIES[pred], CATEGORIES_ITEM[pred_item] ,image_path, CATEGORIES_QC[pred_qc])

                        # # save inference info
                        # current_inference_info = pd.DataFrame([{"timestamp": now_unix,
                        #                                         "image_path": image_path,
                        #                                         "classification_result_type": CATEGORIES[pred],
                        #                                         "classification_result_quantity": CATEGORIES_QC[pred_qc],
                        #                                         "classification_result_item": CATEGORIES_ITEM[pred_item]}])

                        # save_folder = f'./result/{TODAY}/'
                        # if not os.path.exists(save_folder):
                        #     os.makedirs(save_folder)

                        # if not os.path.exists(f'./result/{TODAY}/{now_strf}.csv'):
                        #     current_inference_info.to_csv(f'./result/{TODAY}/{now_strf}.csv', index=False)



                logger.info(f"[*] Disconnected connection from {client_address[0]}:{client_address[1]}")
                client_socket.close()
                
            except KeyboardInterrupt:
                server_socket.close()
                cv2.destroyAllWindows()
                system.destroy_device() ##destroy mv cam
                

                exit()

            except ConnectionResetError:
                logger.info(f"[*] Client Disconnected.")

            except Exception as e:
                logger.error(traceback.format_exc())
                server_socket.close()
                exit()

    system.destroy_device() ##destroy mv cam



















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

















###MV 입도분석 모델 loading 
class granule_sam_test:
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
        self.inference_result_path = '/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/sam_img'

        sam2 = build_sam2(self.model_cfg, self.checkpoint, device = self.DEVICE, apply_postprocessing=False)
        ## For faster inference, we use torch.compile
        self.sam2 = torch.compile(sam2, mode='reduce-overhead')

    def forward_granule(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img)
        # logger.info(f'granule image shape: {img.shape}')
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
        plt.imsave(f"{self.inference_result_path}/{datetime.datetime.now()}_sampyo_sam.jpg", result_image)
        # logger.info(f"SAM result_image final output shape: {result_image.shape}")
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


    def analyze_granule(self, image):
    # def analyze_granule(self, img_path):
        try:

            mm13 = 0
            mm20 = 0
            mm25 = 0
            mm40 = 0
            # image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(image)
            # image = cv2.imread('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/sampyo_sam2.jpg')
            
            # Convert the image to grayscale
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get the mask (predefined threshold)
            mask = cv2.inRange(immg, 1, 255)

            #Find all Objects contour's 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # logger.info(f"contours: {contours}")
            print(contours)
            print(len(contours))
            granule_num_px_list = []
            granule_analysis_dict = {}
            for i in range(len(contours)):
                m=max(contours[i][0])

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

            

            # min_value = min(granule_num_px_list) #0
            # max_value = max(granule_num_px_list) #6201
            # average_value = int(np.round(sum(granule_num_px_list)/len(granule_num_px_list))) ## average ##855
            # median_value = statistics.median(granule_num_px_list) ## 603.5
            # stdev_value = statistics.stdev(granule_num_px_list)
            # num_granule = len(contours) ##1691
            # median_value/25 ##1mm당 pixel 갯수 
            # 94.32*13
            # 94.32*20
            # 94.32*25
            # 94.32*40
            granule_analysis_dict = {"13mm": mm13, '20mm': mm20, '25mm': mm25, '40mm': mm40}
            # logger.info('granule analysis done')

            return granule_analysis_dict
            #return granule_analysis_dict, min_value, max_value, average_value, median_value, num_granule, stdev_value
        except:
            # logger.error(traceback.format_exc())
            return ['None']

new = granule_sam_test()

immg = new.forward_granule("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_2025-02-13 12:00:18.751012_202502130033.jpg")
zzzz = damn.analyze_granule("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/sam_img/2025-02-13 16:14:58.551323_sampyo_sam.jpg")
zzzz = new.analyze_granule(immg)
type(immg)
immg.shape


cv2.imwrite("/home/sdt/Workspace/onvif/python-onvif-zeep/zzzz.jpg",immg)