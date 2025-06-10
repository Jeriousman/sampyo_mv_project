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
from onvif import ONVIFCamera
from torchvision import transforms
import subprocess

#### This is for MaskDINO ###
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

### for MaskDINO setting ###

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

#### This is for depth camera
import depthai as dai
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


SERVER_HOST = cfg['server_host']
SERVER_PORT = 7701



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


if __name__ == "__main__":
    # Fix Seed
    seed_everything(SEED)
    depth_camera = depth_cam()
    print("Depth cam initialized")
    logger.info(f"Depth camera init: {depth_camera}")



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



    # Open Socket Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen()
    logger.info(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")



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
                
                    data = client_socket.recv(1024)  # type: bytes
                    if not data:
                        break

                    print(data)
                    str_data = data.decode('cp949')  # type: str
                    
                    logger.info(f"[*] Received data: {str_data}")
                    vseq_no = str_data.split('|')[1]

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
                        
                        #############DEPTH CAMERA START########################
                        # depth_camera.messages = depth_camera.device.getOutputQueue("rgbd", maxSize=1, blocking=True).tryGetAll()    
                        depth_camera.messages = depth_camera.device.getOutputQueue("rgbd", maxSize=1, blocking=False).getAll()
                        
                        # while True:
                        start = time.time() 

                        if not depth_camera.messages:
                            continue
                        # currentDateAndTime = datetime.datetime.now()
                        logger.info(f"depth_camera.messages: {depth_camera.messages}")
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
                                    with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depth_frame_array_{depth_time}_{item_code}_{type_code}_{from_code}_{item_code}.pickle", 'wb') as f:
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
                                    
                                    cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparity_frame_{disparity_time}_{item_code}_{type_code}_{from_code}_{item_code}.jpg", disparity_frame)
                                    
                                    

                                elif name == 'rgb':
                                    rgb_frame = frame.getCvFrame()##.astype(np.uint16)
                                    current_time_depth_rgb = datetime.datetime.now()
                                    logger.info(f"Depth rgb img saved at : {current_time_depth_rgb}") 
                                    rgb_frame = cv2.resize(rgb_frame, (depth_camera.h, depth_camera.w), interpolation=cv2.INTER_NEAREST)
                                    # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}_{vseq_no}.jpg", rgb_frame)
                                    # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{datetime.datetime.now()}_{vseq_no}.jpg", rgb_frame)
                                    cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{current_time_depth_rgb}_{item_code}_{type_code}_{from_code}_{item_code}.jpg", rgb_frame)                                    
                                
                                    ###### MaskDINO inference ######
                                    # output = mask 
                                    predictions, visualized_output = predictor_vis.run_on_image(rgb_frame)
                                    # combined_mask를 추후 mask로 사용하면 됨 
                                    if len(predictions['instances']) == 2:
                                        combined_mask = torch.any(torch.stack([predictions['instances'][0].pred_masks,predictions['instances'][1].pred_masks]), dim=0)
                                        mask_ck = True
                                    elif len(predictions['instances']) == 1:
                                        combined_mask = predictions['instances'][0].pred_masks
                                        mask_ck = True
                                    elif len(predictions['instances']) == 0:
                                        mask_ck = False
                                    
                                    if mask_ck:
                                        # 결과 확인용 코드 (추후 사용 안해도 됨) 
                                        mask_np = combined_mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255 # 마스크 이미지 저장을 위한 (1, H, W) → (H, W) 변환
                                        cv2.imwrite(f'/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_mask/combined_mask_{current_time_depth_rgb}.jpg', mask_np)
                                        # MaskDINO 결과 이미지 저장 
                                        visualized_output.save(f'/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_mask/masking_result_{current_time_depth_rgb}.jpg')
                                    else:
                                        logger.error("Failed to segment Depth image")

                                ######## MaskDino end ###########
 

                                end = time.time()
                                print(end - start)
                        #############DEPTH CAMERA END########################        



                        from PIL import Image
                        import numpy as np
                        import pickle
                        with open("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depth_frame_array_2025-01-23 11:28:00.708741_32001_B1_1500680000.pickle", "rb") as f:
                            dp = pickle.load(f)

                        with open("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_2025-01-17 13:50:50.056312_202501170082.pickle", "rb") as f:
                            dp = pickle.load(f)


                        with open("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depth_frame_array_2025-02-03 09:32:59.784617_32001_B1_1500680000.pickle", "rb") as f:
                            dp = pickle.load(f)
                    
                        # /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_2025-01-17 13:50:50.040196_202501170082.jpg
                        # /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_2025-01-17 13:50:50.056312_202501170082.pickle
                        # dp[1000:1050,1000:1050]
                        # len(dp[dp > 2000])
                        dp.shape

                        
                        ##값이 59364인것의 인덱스 행렬로 뽑아보아라.
                        ## 316~322,210~332이 덤프트럭 헤드  10 - 3.16 = ?
                        ## 245~280 가 덤프트럭 화물차 위치. 화물칸 아래부분은 125~140까지라고 보자.
                        
                        
                        ##sorting values of depth array
                        sorted_array = np.sort(dp, axis=None)[::-1]#.reshape(array.shape)
                        len(sorted_array)  ##4672512 총 픽셀갯수 
                        sorted_depth = np.sort(np.unique(sorted_array))[::-1]
                        # sorted_array[4400000] ##?????????????
                        # sorted_array[4430000]
                        # sorted_array[4480000]
                        # sorted_array[4480000]
                        dp_ = np.where(dp == 3492, 1, 0)
                        dp_ = np.where(dp == 3492, 1, 0)
                        dp_ = np.where(dp == 6596, 1, 0)
                        dp_ = np.where(dp == 3710, 1, 0)
                        dp_ = np.where(dp == 3958, 1, 0)
                        dp_ = np.where(dp == 4240, 1, 0)
                        dp_ = np.where(dp == 4566, 1, 0)
                        dp_ = np.where(dp == 4947, 1, 0)
                        dp_ = np.where(dp == 5397, 1, 0)
                        dp_ = np.where(dp == 5936, 1, 0)
                        dp_ = np.where(dp == 6596, 1, 0) ##contaier
                        dp_ = np.where(dp == 7421, 1, 0) ##contaier
                        dp_ = np.where(dp == 8481, 1, 0) ##
                        dp_ = np.where(dp == 9894, 1, 0) ##바닥 
                        dp_ = np.where(dp == 11873, 1, 0)
                        dp_ = np.where(dp == 14841, 1, 0)
                        dp_ = np.where(dp == 19788, 1, 0)
                        dp_ = np.where(dp == 29682, 1, 0)
                        dp_ = np.where(dp == 59364, 1, 0)
                        import matplotlib.pyplot as plt
                        plt.imshow(dp_)
                        9894 - dp_
                        dp_
                        11873 - 3492
                        indices = np.where(dp == 59364)
                        indices[0]
                        indices[1]
                        
                        indices = np.where(dp > 9000)
                        indices[0]
                        indices[1]
                        len(indices[0])
                        dp[indices[0], indices[1]] = 1

                        indices = np.where(dp == 0)
                        indices[0]
                        indices[1]
                        len(indices[0])

                        dp
                        dp[] = 1

                        dp[158, 467]
                        len(dp[dp < 2000])  ## probably the 
                        np.where(dp[(dp > 8100) & (dp < 8500)], 0, 1)
                        len(dp[(dp > 8100) & (dp < 8500)])
                        len(dp[dp > 8500])
                        
                        
                        
                        dp.shape
                        1872*2496
                        np.max(dp)
                        dp[(dp > 8100) and (dp < 8500)]
                        8100~ 8500
                        10-1.8
                        dp.shape
                        max(dp)
                        depth_array =  
                        pil_img_rgb = Image.open('/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_2025-01-11 15:37:59.621154.jpg')
                        np.asarray(pil_img_depth)
                        np.asarray(pil_img_rgb)
                        rgb_frame * depth_frame







                        # mydb.insert(str_data, vin_date, CATEGORIES[pred], image_path)
                        sdt_msdb.upsert_depth(str_data, quantity)




                logger.info(f"[*] Disconnected connection from {client_address[0]}:{client_address[1]}")
                client_socket.close()
                
            except KeyboardInterrupt:
                server_socket.close()
                cv2.destroyAllWindows()
                

                exit()

            except ConnectionResetError:
                logger.info(f"[*] Client Disconnected.")

            except Exception as e:
                logger.error(traceback.format_exc())
                server_socket.close()
                exit()





















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