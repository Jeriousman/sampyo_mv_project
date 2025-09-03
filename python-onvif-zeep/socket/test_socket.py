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
from onvif import ONVIFCamera
from torchvision import transforms

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
        filename=f"./logs/test.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.addHandler(log_fileHandler)

######################################################################
#                              Config                                #
######################################################################
with open('./config/server_info.json', 'r') as f:
    cfg = json.load(f)

SEED = cfg['model']['seed']
MODEL_NAME = cfg['model']['name']
NUM_CLASSES = cfg['model']['num_classes']
DEVICE = cfg['model']['device']
MODEL_CKPT = cfg['model']['ckpt_path']
CATEGORIES = {0: '모래',
              1: '자갈',
              2: '덮개',
              3: '빈차',
              4: '레미콘',
              5: '차량없음'}

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
with open('./config/db_info.json', 'r') as f:
    db_info = json.load(f)


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
        else:
            logger.info("Failed to fetch snapshot")

        return os.path.abspath(image_path)


######################################################################
#                              Model                                 #
######################################################################
class Model:
    def __init__(self, ckpt_path):
        self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        self.transform = transforms.Compose([transforms.Resize((384, 384)),
                                             transforms.ToTensor()])
        self.inference('./dummy.jpg')

    def inference(self, image_path):
        image = Image.open(image_path)
        t_image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            self.model.eval()

            inputs = t_image.to(DEVICE)
            outputs = self.model(inputs)

            preds = torch.argmax(outputs, dim=-1)

        return preds.detach().cpu().numpy()[0]


if __name__ == "__main__":
    # Fix Seed
    seed_everything(SEED)

    # Load Camera
    onvif_cam = ONVIFCam()

    # MSSQL DB Connector
    sampyo_msdb = mssql.DB(host=SAMPYO_HOST,
                           dbname=SAMPYO_DBNAME,
                           username=SAMPYO_USERNAME,
                           password=SAMPYO_PASSWORD)

    sdt_msdb = mssql.DB(host=SDT_HOST,
                        dbname=SDT_DBNAME,
                        username=SDT_USERNAME,
                        password=SDT_PASSWORD)

    # Load Model
    model = Model(MODEL_CKPT)
    
    # Open Socket Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    server_socket.bind((SERVER_HOST, 7800))
    server_socket.listen()
    logger.info(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")

    while True:
        try:
            client_socket, client_address = server_socket.accept()
            logger.info(f"[*] Accepted connection from {client_address[0]}:{client_address[1]}")

            while True:
                initialize_vseq_no()  # Initialize dictionary for data duplication check.
                
                data = client_socket.recv(1024)  # type: bytes
                if not data:
                    break

                str_data = data.decode('utf-8')  # type: str
                logger.info(f"[*] Received data: {str_data}")
                vseq_no = str_data.split('|')[1]

                vin_date = datetime.datetime.now()
                if not is_duplicated(vseq_no):
                    now_datetime = datetime.datetime.now()
                    now_unix = int(now_datetime.timestamp())
                    now_strf = now_datetime.strftime("%Y%m%d-%H%M%S")

                    image_path = onvif_cam.get_snapshot_and_save(now_strf)
                    pred = model.inference(image_path)

                    # update inference result to DB
                    sampyo_msdb.update(str_data, CATEGORIES[pred])
                    # mydb.insert(str_data, vin_date, CATEGORIES[pred], image_path)
                    sdt_msdb.insert(str_data, vin_date, CATEGORIES[pred], image_path)

                    # save inference info
                    current_inference_info = pd.DataFrame([{"timestamp": now_unix,
                                                            "image_path": image_path,
                                                            "classification_result": CATEGORIES[pred]}])

                    save_folder = f'./result/{TODAY}/'
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    if not os.path.exists(f'./result/{TODAY}/{now_strf}.csv'):
                        current_inference_info.to_csv(f'./result/{TODAY}/{now_strf}.csv', index=False)

            logger.info(f"[*] Disconnected connection from {client_address[0]}:{client_address[1]}")
            client_socket.close()
            
        except KeyboardInterrupt:
            server_socket.close()
            exit()

        except ConnectionResetError:
            logger.info(f"[*] Client Disconnected.")

        except Exception as e:
            logger.error(traceback.format_exc())
            server_socket.close()
            exit()

