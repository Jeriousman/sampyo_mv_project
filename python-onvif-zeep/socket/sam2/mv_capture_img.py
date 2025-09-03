

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
import threading
import traceback
import requests
from requests.auth import HTTPDigestAuth
import statistics
# import logging.handlers
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
import logging
import pickle
sys.path.append("/home/sdt/Workspace/onvif/python-onvif-zeep/socket")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



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


seq_no_dict = {}
TODAY = datetime.date.today()
YESTERDAY = TODAY - timedelta(days=1)




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




class MVCam:
    def __init__(:
        try:
            device_infos = system.device_infos ## 연결된 루시드 MV 카메라 정보 호출 
            devices = create_devices_with_tries() ## MV카메라 장치를 발견하고, 발견되면 create_device()를 호출하여 장치 객체를 생성
            device = system.select_device(devices) ## 발견된 장치 객체 중에 사용할 디바이스 선택         

            ########## added by dykim ##########
            # Get device stream nodemap
            tl_stream_nodemap = device.tl_stream_nodemap ## 스트림 설정을 위한 노드맵
            # Enable stream auto negotiate packet size 
            tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True # 노드맵 안에서 이미지를 전송할 때 자동으로 패킷 크기를 협상하도록 설정 
            # Enable stream packet resend 
            tl_stream_nodemap['StreamPacketResendEnable'].value = True ## 루시드 카메라가 PC로 이미지를 전송할 때 손실된 패킷을 자동으로 재전송 하도록 설정 
            # Get/Set nodes -----------------------------------------------------------
            #nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])
            ##dir(device.nodemap)
            ##device.nodemap.feature_names
            ##nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat',  'ExposureAuto', 'ExposureTime','ExposureAutoLimitAuto', 'ExposureAutoLowerLimit ','ExposureAutoUpperLimit', 'GainAuto', 'Gain', 'DeviceStreamChannelPacketSize','OffsetX','OffsetY',])
            # nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat',  'ExposureAuto', 'ExposureTime','ExposureAutoLimitAuto', 'ExposureAutoUpperLimit', 'ExposureAutoLowerLimit','GainAuto', 'Gain', 'DeviceStreamChannelPacketSize','OffsetX','OffsetY',])
            # ##nodes
            # ############################################################################################################
            # ############################################################################################################
            # ###카메라 밝기/모션블러 컨트롤 관련 
            # #nodes['ExposureAuto'].value = 'Off' ## GenCam 기반 카메라에서 자동 노출 기능을 끄는 설정 
            # ##nodes['ExposureTime'].value = 11000.0 ##7500.0 ##2250.0 ##1000.0 # 5000.0 ## 자동 노출을 off 하였으니, 노출 시간을 수동으로 설정 해 주는 것. 노출시간이 짧으면 이미지가 어두울 것이지만 모션블러는 적을것이다.
            # nodes['ExposureAuto'].value = 'Continuous' ## GenCam 기반 카메라에서 계속적인 자동 노출 설정 
            # nodes['ExposureAutoLimitAuto'].value = 'Off'
            # nodes['GainAuto'].value = 'Continuous'  ## 필요 없으면 'Off'
            # nodes['ExposureAutoUpperLimit'].value = 8000.0  
            # nodes['ExposureAutoLowerLimit'].value = 2250.0  
            # ##nodes['Gain'].value
            ############################################################################################################
            # logger.info(f"device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")


            ## 호준 업데이트 
            nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat',  'ExposureAuto', 'ExposureTime','ExposureAutoLimitAuto', 'ExposureAutoUpperLimit', 'ExposureAutoLowerLimit','GainAuto', 'Gain', 'DeviceStreamChannelPacketSize','OffsetX','OffsetY',])
            
            
            

            ############################################################################################################
            ############################################################################################################
            ###카메라 밝기/모션블러 컨트롤 관련 (호준 업데이트 )
            #nodes['ExposureAuto'].value = 'Off' ## GenCam 기반 카메라에서 자동 노출 기능을 끄는 설정 
            ##nodes['ExposureTime'].value = 11000.0 ##7500.0 ##2250.0 ##1000.0 # 5000.0 ## 자동 노출을 off 하였으니, 노출 시간을 수동으로 설정 해 주는 것. 노출시간이 짧으면 이미지가 어두울 것이지만 모션블러는 적을것이다.
            nodes['ExposureAuto'].value = 'Continuous' ## GenCam 기반 카메라에서 계속적인 자동 노출 설정 
            nodes['ExposureAutoLimitAuto'].value = 'Off'

            nodes['GainAuto'].value = 'Off'  ## 필요 없으면 'Off'
            nodes['ExposureAutoUpperLimit'].value = 1500.0  
            nodes['ExposureAutoLowerLimit'].value = 1000.0  
            ##ExposureAuto를 최대한 줄이고, gain을 늘려라 
            nodes['GainAuto'].value = 'Off' ##"continuous"

            # 필요 시 Gain 수동 설정 (대신 GainAuto = 'Off' 해야 함)
            # nodes['GainAuto'].value = 'Off'
            nodes['Gain'].value = 30.0


            ##nodes['Gain'].value
            logger.info(f"device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}, GainAuto: {nodes['GainAuto'].value}")
            ############################################################################################################







            nodes['OffsetX'].value = 0
            nodes['OffsetY'].value = 0
            # logger.info(f"device.nodemap - OffsetX, OffsetY : {nodes['OffsetX'].value}, {nodes['OffsetY'].value}")
        
            # to acquisition rapid, packet size setting: origin = 1500
            stream_packet_size_max = nodes['DeviceStreamChannelPacketSize'].max ## 현재 카메라에서 전송 가능한 최대 패킷 크기(MTU)에 대한 값을 가져오는 것 
            nodes['DeviceStreamChannelPacketSize'].value = stream_packet_size_max ## 패킷 사이즈를 최대 값으로 설정 
            # logger.info(f"device.nodemap - DeviceStreamChannelPacketSize, packet size max : {nodes['DeviceStreamChannelPacketSize'].value}, {nodes['DeviceStreamChannelPacketSize'].max}")
        
        
            #print(f"device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
            # set exposure time: origin=16394.6
            # logger.info(f"nodes['ExposureTime'].min: {nodes['ExposureTime'].min}")
            
            #print(f"device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
            # Set pixel format to Mono8
            pixel_format_name = 'BayerRG8'
            #logger.info(f'Setting Pixel Format to {pixel_format_name}')
            nodes['PixelFormat'].value = pixel_format_name
            
            # setting width, height
            #print('Setting Width to 2700')
            nodes['Width'].value = 2700 ## 이미지의 해상도 폭 설정
            #print('Setting Height to 2000')
            height = nodes['Height']
            height.value = 2000 ## 이미지의 해상도 높이 설정 

            # demonstrages save: mono8 to png
            #device.start_stream(1)
            #logger.info('Started stream')
        except:
            # logger.error(traceback.format_exc())
            #device.stop_stream()
            system.destroy_device()
            # logger.error(traceback.format_exc())
            # logger.error('Destroyed all created devices')
           



    def capture(:
    # def capture(:
        try:
            ### added by dykim ###
            device.start_stream(1) ##이미지를 버퍼에 저장하는데, 버퍼에 1개의 프레임만 밀어 넣으라는 것 
            # logger.info('Started stream')
            # logger.info('Grabbing an image buffer')
            image_buffer = device.get_buffer() ## 버퍼에 밀어넣어진 1개의 이미지 프레임을 가져오는 함수 
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
               image_buffer.pdata,(image_buffer.height, image_buffer.width))  ## C 라이브러리 기반 SDK에서 이미지를 포인터 형태로 전송, 그 후 파이썬에서 다루기 위해 numpy array로 변환.
            
            # rgb_img = cv2.cvtColor(nparray_reshaped, cv2.COLOR_BayerRG2BGR) ## RGB이미지를 BGR로 변환 
            rgb_img = cv2.cvtColor(nparray_reshaped, cv2.COLOR_BayerBG2RGB) ## RGB이미지를 BGR로 변환 
            # mv_img_array = Image.fromarray(nparray_reshaped)
            now = datetime.datetime.now()
            now = now.strftime('%Y-%m-%d %H:%M:%S')
            cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/mv_size_measure_imgs/img_{now}.png", nparray_reshaped)
            # mv_img_array.save(f"/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/mv_size_measure_imgs/img_{now}.png")



            
            device.requeue_buffer(image_buffer)
            #print('requeue buffer')
            device.stop_stream()
            #print('stop stream after requeue buffer')
            return rgb_img
        
        except Exception as e:
            device.stop_stream()
            #print(e)
            # logger.info(traceback.format_exc())
        


if __name__ == "__main__":
    # Fix Seed


    # Load Camera
    mv_cam = MVCam()
    mv_cam
    print("MV cam initialized")
    # logger.info(f"MV camera initialized: {mv_cam}")


    rgb_img=mv_cam.capture() ##이것이 MV camera 이미지를 캡쳐하는 함수이다 
    plt.imshow(rgb_img)
    type(rgb_img)
    type(rgb_img)
    values = mv_cam.device.nodemap.get_node(['Width', 'Height', 'PixelFormat',  'ExposureAuto', 'ExposureTime','DeviceStreamChannelPacketSize','OffsetX','OffsetY',])
    values['ExposureTime']
    rgb_img




    #### 카메라 설정하기 위한 코드 
    device_infos = system.device_infos ## 연결된 루시드 MV 카메라 정보 호출 
    devices = create_devices_with_tries() ## MV카메라 장치를 발견하고, 발견되면 create_device()를 호출하여 장치 객체를 생성
    device = system.select_device(devices) ## 발견된 장치 객체 중에 사용할 디바이스 선택         

    ########## added by dykim ##########
    # Get device stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap ## 스트림 설정을 위한 노드맵
    # Enable stream auto negotiate packet size 
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True # 노드맵 안에서 이미지를 전송할 때 자동으로 패킷 크기를 협상하도록 설정 
    # Enable stream packet resend 
    tl_stream_nodemap['StreamPacketResendEnable'].value = True ## 루시드 카메라가 PC로 이미지를 전송할 때 손실된 패킷을 자동으로 재전송 하도록 설정 
    # Get/Set nodes -----------------------------------------------------------
    #nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])
    nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat',  'ExposureAuto', 'ExposureTime','DeviceStreamChannelPacketSize','OffsetX','OffsetY',])
    nodes['ExposureAuto'].value = 'Off' ## GenCam 기반 카메라에서 자동 노출 기능을 끄는 설정 
    # logger.info(f"device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
    
    nodes['OffsetX'].value = 0
    nodes['OffsetY'].value = 0
    # logger.info(f"device.nodemap - OffsetX, OffsetY : {nodes['OffsetX'].value}, {nodes['OffsetY'].value}")

    # to acquisition rapid, packet size setting: origin = 1500
    stream_packet_size_max = nodes['DeviceStreamChannelPacketSize'].max ## 현재 카메라에서 전송 가능한 최대 패킷 크기(MTU)에 대한 값을 가져오는 것 
    nodes['DeviceStreamChannelPacketSize'].value = stream_packet_size_max ## 패킷 사이즈를 최대 값으로 설정 
    # logger.info(f"device.nodemap - DeviceStreamChannelPacketSize, packet size max : {nodes['DeviceStreamChannelPacketSize'].value}, {nodes['DeviceStreamChannelPacketSize'].max}")


    #print(f"device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
    # set exposure time: origin=16394.6
    # logger.info(f"nodes['ExposureTime'].min: {nodes['ExposureTime'].min}")
    nodes['ExposureTime'].value = 2250.0 ##7500.0 ##2250.0 ##1000.0 # 5000.0 ## 자동 노출을 off 하였으니, 노출 시간을 수동으로 설정 해 주는 것. 노출시간이 짧으면 이미지가 어두울 것이지만 모션블러는 적을것이다.
    #print(f"device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
    # Set pixel format to Mono8
    pixel_format_name = 'BayerRG8'
    #logger.info(f'Setting Pixel Format to {pixel_format_name}')
    nodes['PixelFormat'].value = pixel_format_name
    
    # setting width, height
    #print('Setting Width to 2700')
    nodes['Width'].value = 2700 ## 이미지의 해상도 폭 설정
    #print('Setting Height to 2000')
    height = nodes['Height']
    height.value = 2000 ## 이미지의 해상도 높이 설정 














