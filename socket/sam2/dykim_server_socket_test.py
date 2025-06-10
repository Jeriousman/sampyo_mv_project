import sys
print(sys.prefix)
import os
import io
import time
import json
#import base64
import random
#import socket
#import pyodbc
import datetime
from datetime import timedelta

import traceback
#import requests
#from requests.auth import HTTPDigestAuth

import logging.handlers

from PIL import Image

#import timm
#import torch
#import pyodbc
import numpy as np
#import pandas as pd
import cv2 
#from onvif import ONVIFCamera
#from torchvision import transforms





# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from arena_api.system import system
import subprocess


#### This is for depth camera
import depthai as dai
from datetime import timedelta

import pickle
#sys.path.append("/home/sdt/Workspace/onvif/python-onvif-zeep/socket")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from utils import mssql

######################################################################
#                             Save Log                               #
######################################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
        filename=f"./socket_teat.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.addHandler(log_fileHandler)

######################################################################
#                              Config                                #
######################################################################
# with open('./config/server_info.json', 'r') as f:
# with open('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/config/server_info.json', 'r') as f:
#     cfg = json.load(f)

# SEED = cfg['model']['seed']
# MODEL_NAME = cfg['model']['name']
# NUM_CLASSES = cfg['model']['num_classes']
# NUM_CLASSES_QC = cfg['model']['num_classes_qc']
# NUM_CLASSES_ITEM = cfg['model']['num_classes_item']
# DEVICE = cfg['model']['device']
# MODEL_CKPT = cfg['model']['ckpt_path']
# MODEL_CKPT_QC = cfg['model']['ckpt_path_qc']
# MODEL_CKPT_ITEM = cfg['model']['ckpt_path_item']
# CATEGORIES = {0: '모래',
#               1: '자갈',
#               2: '덮개',
#               3: '빈차',
#               4: '레미콘',
#               5: '차량없음'}

# CATEGORIES_QC = {0: '부족',
#                  1: '정상'}

# CATEGORIES_ITEM = {0: '석산',
#                    1: '석산',
#                    2: '발파석',
#                    3: '발파석'}

# IMAGE_BUCKET = cfg['image_bucket']

# SERVER_HOST = cfg['server_host']
# SERVER_PORT = cfg['server_port']

# CAM_IP = cfg['cam_ip']
# CAM_PORT = cfg['cam_port']
# CAM_ID = cfg['cam_id']
# CAM_PW = cfg['cam_pw']


# seq_no_dict = {}
# TODAY = datetime.date.today()
# YESTERDAY = TODAY - timedelta(days=1)

######################################################################
#                              Config                                #
######################################################################
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


# ######################################################################
#                             Fix seed                               #
######################################################################
# def seed_everything(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if use multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)


######################################################################
#                   Check duplication in veqs_no                     #
######################################################################
# def is_duplicated(vseq_no):
#     global seq_no_dict

#     if vseq_no in seq_no_dict:
#         return True
#     else:
#         seq_no_dict[vseq_no] = 1
#         return False


######################################################################
#            Initialize dictionary for check duplication             #
######################################################################
# def initialize_vseq_no():
#     global sqe_no_dict, TODAY, YESTERDAY

#     TODAY = datetime.date.today()
#     if str(YESTERDAY) != str(TODAY):
#         YESTERDAY = TODAY
#         seq_no_dict = {}
#         logger.info(f'Initialize vseq_no dictionary.')


######################################################################
#            Initialize dictionary for check duplication             #
######################################################################
# def encode_image(image_path):
#     with open(image_path, 'rb') as f:
#         # load image
#         image = Image.open(image_path)
        
#         # resize image
#         width, height = image.size
#         resized_image = image.resize((width // 10, height // 10))

#         io_buffer = io.BytesIO()
#         resized_image.save(io_buffer, format='JPEG')
        
#         # encode image
#         encoded_image = base64.b64encode(io_buffer.getvalue())

#     return encoded_image.decode()


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
# class ONVIFCam:
#     def __init__(self):
#         self.cam = ONVIFCamera(CAM_IP, CAM_PORT, CAM_ID, CAM_PW)
#         self.media_service = self.cam.create_media_service()
#         self.media_profile = self.media_service.GetProfiles()[0]

#     def get_snapshot_and_save(self, now_strf):
#         snapshot_uri = self.media_service.GetSnapshotUri({'ProfileToken': self.media_profile.token}).Uri
#         response = requests.get(snapshot_uri, auth=HTTPDigestAuth(CAM_ID, CAM_PW))

#         if response.status_code == 200:
#             file_path = f'{now_strf}.jpg'
#             save_path = os.path.join(IMAGE_BUCKET, str(TODAY))

#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
             
#             image_path = os.path.join(save_path, file_path)
#             with open(image_path, 'wb') as f:
#                 for chunk in response.iter_content(1024):
#                     f.write(chunk)
#         else:
#             logger.info("Failed to fetch snapshot")

#         return os.path.abspath(image_path)






# class depth_cam:
#     def __init__(self):
#         # Create pipeline
#         self.pipeline = dai.Pipeline()
#         self.device = dai.Device()
#         self.fps = 30

#         # The disparity is computed at this resolution, then upscaled to RGB resolution
#         #monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
#         self.monoResolution = dai.MonoCameraProperties.SensorResolution.THE_800_P
#         # Define sources and outputs
#         self.camRgb = self.pipeline.create(dai.node.ColorCamera)
#         self.left = self.pipeline.create(dai.node.MonoCamera)
#         self.right = self.pipeline.create(dai.node.MonoCamera)
#         self.stereo = self.pipeline.create(dai.node.StereoDepth)

#         try:
#             calibData = self.device.readCalibration2()
#             lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
#             if lensPosition:
#                 self.camRgb.initialControl.setManualFocus(lensPosition)
#         except:
#             raise

#         #Properties
#         self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
#         #camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#         self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP) # 4056x3040
#         self.camRgb.setFps(self.fps)

#         self.left.setResolution(self.monoResolution)
#         self.left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
#         self.left.setFps(self.fps)
#         self.right.setResolution(self.monoResolution)
#         self.right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
#         self.right.setFps(self.fps)

#         self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
#         # LR-check is required for depth alignment
#         self.stereo.setLeftRightCheck(True)
#         self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
#         self.h, self.w = 1248*2, 936*2
#         # # 4056x3040
#         self.stereo.setOutputSize(self.h, self.w)
#         #stereo.setOutputSize(1280, 720)
#         # stereo.setOutputSize(4056, 3040)
#         self.sync = self.pipeline.create(dai.node.Sync)
#         self.sync.setSyncThreshold(timedelta(milliseconds=70))

#         # Linking
#         self.camRgb.isp.link(self.sync.inputs["rgb"])
#         self.left.out.link(self.stereo.left)
#         self.right.out.link(self.stereo.right)
#         self.stereo.depth.link(self.sync.inputs["depth"])

#         self.sync_out = self.pipeline.createXLinkOut()
#         self.sync_out.setStreamName("rgbd")
#         self.sync.out.link(self.sync_out.input)




class MVCam:
    def __init__(self):
                
        self.device_infos = system.device_infos
        self.device = system.create_device(device_infos=self.device_infos[0])[0]
        self.device.start_stream()

    # def capture(self, vseq_no):
    def capture(self):
        try:
            
            image_buffer = self.device.get_buffer()
            current_time_mv = datetime.datetime.now()
            logger.info(f"MV rgb img saved at : {current_time_mv}") 
            ##ctypeslib is a module that makes C library run in python. ##pdata = pixel data?
            ##(3000, 4096, 1) shape
            np_img = np.ctypeslib.as_array(image_buffer.pdata,shape=(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))
            ##(3000, 4096, 3) shape. Grayscale to RGB 
            rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BayerRG2RGB)
            # rgb_img.shape
            #cv2.imshow('lucid_img', rgb_img)
            # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/rgb_frame_{current_time_mv}_{vseq_no}.jpg", rgb_img)
            #cv2.imwrite(f"./imgs/rgb_frame_{current_time_mv}.jpg", rgb_img)
            cv2.imwrite(f"./imgs/rgb_frame_test.jpg", rgb_img)
            
            
            self.device.requeue_buffer(image_buffer)
            # print('RGB: ', rgb_img.shape)
            return rgb_img
        
        except Exception as e:
            print(e)
            logger.info(traceback.format_exc())
            self.device.stop_stream()


######################################################################
#                              Model                                 #
######################################################################
# class Model:
#     def __init__(self, ckpt_path, num_classes):
#         self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes).to(DEVICE)
#         self.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        
#         self.transform = transforms.Compose([transforms.Resize((384, 384)),
#                                              transforms.ToTensor()])
#         self.inference('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/dummy.jpg')

#     def inference(self, image_path):
#         image = Image.open(image_path)
#         t_image = self.transform(image).unsqueeze(0)

#         with torch.no_grad():
#             self.model.eval()

#             inputs = t_image.to(DEVICE)
#             outputs = self.model(inputs)

#             preds = torch.argmax(outputs, dim=-1)

#         return preds.detach().cpu().numpy()[0]


if __name__ == "__main__":
    # Fix Seed
    #seed_everything(SEED)

    # Load Camera
    #onvif_cam = ONVIFCam()
    #logger.info(f"onvif_cam initialized: {onvif_cam}")
    # depth_camera = depth_cam()
    # print("Depth cam initialized")
    # logger.info(f"Depth camera init: {depth_camera}")
    mv_cam = MVCam()
    print("MV cam initialized")
    logger.info(f"MV camera init: {mv_cam}")


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

    # # Load Model
    # model = Model(MODEL_CKPT,NUM_CLASSES)
    # logger.info('SDT model loaded')
    # model_qc = Model(MODEL_CKPT_QC, NUM_CLASSES_QC)
    # model_item = Model(MODEL_CKPT_ITEM, NUM_CLASSES_ITEM)

    # # Open Socket Server
    # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    # server_socket.bind((SERVER_HOST, SERVER_PORT))
    # server_socket.listen()
    # logger.info(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")



    # with depth_camera.device:
    #     depth_camera.device.startPipeline(depth_camera.pipeline)

    while True:
        try:
            #client_socket, client_address = server_socket.accept()
            #logger.info(f"[*] Accepted connection from {client_address[0]}:{client_address[1]}")

        # Connect to device and start pipeline
        # with depth_camera.device:
        #     depth_camera.device.startPipeline(depth_camera.pipeline)
            while True:
                #initialize_vseq_no()  # Initialize dictionary for data duplication check.
            
                #data = client_socket.recv(1024)  # type: bytes
                #if not data:
                #    break

                # print(data)
                # str_data = data.decode('cp949')  # type: str
                
                # logger.info(f"[*] Received data: {str_data}")
                # vseq_no = str_data.split('|')[1]

                # vin_date = datetime.datetime.now()
                # if not is_duplicated(vseq_no):

                #     now_datetime = datetime.datetime.now()
                #     logger.info(f"sensor trigger received at {now_datetime}")
                #     now_unix = int(now_datetime.timestamp())
                #     now_strf = now_datetime.strftime("%Y%m%d-%H%M%S")
                #     logger.info(f"Trigger type: {str_data.split('|')[-1][0]}")

                #     if len(str_data[1:-1].split('|')) == 7:
                #         vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url = str_data[1:-1].split('|')

                #     elif len(str_data[1:-1].split('|')) == 8:
                #         vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url, _ = str_data[1:-1].split('|')

                #     ##sdt_msdb.categories['item']['31013']        
                #     if item_code not in ["31013", "31019", "31025", "31067", "32001"]:
                #         continue

                #     elif type_code not in ["S1", "B1", "S2"]:
                #         continue

                #     # sdt_msdb.categories['from']['31013']
                #     elif from_code not in ["0493870000", "0669560000", "0712360000",
                #                 "0736040000", "0791340000", "0892200000", "0935600000",
                #                 "1326700000", "1500680000", "1512120000", "1512730000",
                #                 "1920700000", "2298820000", "RC10017"]:
                #         continue
                    # test                        
                    rgb_img=mv_cam.capture() ##이것이 이미지를 캡쳐하는 함수이다 

                    # # logger.info(rgb_img) 
                    #depth_camera.messages = depth_camera.device.getOutputQueue("rgbd", maxSize=1).tryGetAll()    
                    # while True:
                    # start = time.time() 

                    # if not depth_camera.messages:
                    #     continue
                    # currentDateAndTime = datetime.datetime.now()
                    
                    # for message_group in depth_camera.messages:
                        # for name, frame in message_group:
                        #     print(f"{name}: {frame.getSequenceNum()}")

                            # if name == 'depth':
                            #     # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                            #     depth_frame = frame.getCvFrame()##.astype(np.uint16)
                            #     current_time_depth_depth = datetime.datetime.now()
                            #     logger.info(f"Depth depth img saved at : {current_time_depth_depth}") 
                                # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}_{vseq_no}.jpg", depth_frame)
                                # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}.jpg", depth_frame)
                                # depth_time = datetime.datetime.now()
                                # cv2.imwrite(f"./imgs/depthFrame_{depth_time}_{vseq_no}.jpg", depth_frame)
                                # print('depth iamge collected')
                                #with open(f"./imgs/depthFrameArray_{depth_time}_{vseq_no}.pickle", 'wb') as f:
                                #    pickle.dump(depth_frame, f)
                            ###########################################################################################
                            ## uncomment this for depth+RGB blending
                                # depth_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                                # depth_frame = cv2.equalizeHist(depth_frame)
                                # depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)
                            ###########################################################################################
                                # print('depth: ', depth_frame)
                                # print('depth: ', depth_frame.shape)

                            # elif name == 'disparity':
                            #     disparity_frame = frame.getCvFrame()
                            #     cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparity_frame_{currentDateAndTime}.jpg", disparity_frame)

                            # elif name == 'rgb':
                            #     rgb_frame = frame.getCvFrame()##.astype(np.uint16)
                            #     current_time_depth_rgb = datetime.datetime.now()
                            #     logger.info(f"Depth rgb img saved at : {current_time_depth_rgb}") 
                            #     rgb_frame = cv2.resize(rgb_frame, (depth_camera.h, depth_camera.w), interpolation=cv2.INTER_NEAREST)
                            #     # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}_{vseq_no}.jpg", rgb_frame)
                            #     # cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}.jpg", rgb_frame)
                            #     cv2.imwrite(f"./imgs/rgb_frame_{datetime.datetime.now()}_{vseq_no}.jpg", rgb_frame)
                            #     # print('RGB: ', rgb_frame)
                            #     # print('RGB iamge collected')
                            #     # print('RGB: ', rgb_frame.shape)
                    
                            # end = time.time()
                            # print(end - start)
                            

            #         image_path = onvif_cam.get_snapshot_and_save(now_strf)
            #         #=======================================
            #         pred = model.inference(image_path)
            #         pred_qc = model_qc.inference(image_path)
            #         pred_item = model_item.inference(image_path)
            #         # update inference result to DB
            #         sampyo_msdb.update(str_data, CATEGORIES[pred])
            #         # mydb.insert(str_data, vin_date, CATEGORIES[pred], image_path)
            #         sdt_msdb.insert(str_data, vin_date, CATEGORIES[pred], CATEGORIES_ITEM[pred_item] ,image_path, CATEGORIES_QC[pred_qc])

            #         # save inference info
            #         current_inference_info = pd.DataFrame([{"timestamp": now_unix,
            #                                                 "image_path": image_path,
            #                                                 "classification_result_type": CATEGORIES[pred],
            #                                                 "classification_result_quantity": CATEGORIES_QC[pred_qc],
            #                                                 "classification_result_item": CATEGORIES_ITEM[pred_item]}])

            #         save_folder = f'./result/{TODAY}/'
            #         if not os.path.exists(save_folder):
            #             os.makedirs(save_folder)

            #         if not os.path.exists(f'./result/{TODAY}/{now_strf}.csv'):
            #             current_inference_info.to_csv(f'./result/{TODAY}/{now_strf}.csv', index=False)



            # logger.info(f"[*] Disconnected connection from {client_address[0]}:{client_address[1]}")
            # client_socket.close()
            
        except KeyboardInterrupt:
            #server_socket.close()
            cv2.destroyAllWindows()
            system.destroy_device() ##destroy mv cam
            

            exit()

        #except ConnectionResetError:
        #    logger.info(f"[*] Client Disconnected.")

        except Exception as e:
            logger.error(traceback.format_exc())
            #server_socket.close()
            exit()

    system.destroy_device() ##destroy mv cam


# pil_img_rgb = Image.open('/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_2025-01-11 15:37:59.621154.jpg')

