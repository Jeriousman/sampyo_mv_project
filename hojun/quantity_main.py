
#############
##Import libraries
 
## For depth camera
import cv2
import depthai as dai
import numpy as np
import pickle
import time

##for MaskDINO
import sys
import os

"""
from MaskDINO import train_net
import train_net


"""

import depthai as dai
## getting camera intrinsics 
## https://docs.luxonis.com/software/depthai/examples/calibration_reader/

with dai.Device() as device:
  calibData = device.readCalibration()
  RIGHT_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT) ##CAM_C
  RGB_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB) ##CAM_A
  LEFT_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT) ##CAM_B
  print("RGB intrinsics: ", RGB_intrinsics)
  print("RGB intrinsics type: ", type(RGB_intrinsics))
#   print('Right mono camera focal length in pixels:', intrinsics[0][0])


# [[3070.4697265625, 0.0,               1884.3509521484375], 
#  [0.0,             3069.542236328125, 1112.5499267578125], 
#  [0.0,             0.0,               1.0]]


################
## setting parameters








###############
## Getting triger signal from the sensor












# #######################
# ## Connect Depth Camera and get RGB & depth images

# import cv2
# import depthai as dai
# import numpy as np
# import pickle

# def getDisparityFrame(frame, cvColorMap):
#     maxDisp = stereo.initialConfig.getMaxDisparity()
#     disp = (frame * (255.0 / maxDisp)).astype(np.uint8)
#     disp = cv2.applyColorMap(disp, cvColorMap)

#     return disp
# cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
# cvColorMap[0] = [0, 0, 0]

# # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
# extended_disparity = False
# # Better accuracy for longer distance, fractional disparity 32-levels:
# subpixel = False
# # Better handling for occlusions:
# lr_check = False

# # Create pipeline
# pipeline = dai.Pipeline()

# # Define sources and outputs
# monoLeft = pipeline.create(dai.node.MonoCamera)
# monoRight = pipeline.create(dai.node.MonoCamera)
# stereo = pipeline.create(dai.node.StereoDepth) ##This catches disparity and depth.
# ##these below are for rgb
# # Define source and output
# camRgb = pipeline.create(dai.node.ColorCamera)
# xoutRgb = pipeline.create(dai.node.XLinkOut)
# xoutRgb.setStreamName("rgb")


# # Properties
# camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setVideoSize(1920, 1080)

# xoutRgb.input.setBlocking(False)
# xoutRgb.input.setQueueSize(1)

# # Linking
# camRgb.video.link(xoutRgb.input)
# xoutDisparity = pipeline.create(dai.node.XLinkOut) ##XLink sends to the host (computer)
# xoutDepth = pipeline.create(dai.node.XLinkOut)
# xoutDisparity.setStreamName("disparity")
# xoutDepth.setStreamName("depth")
# #dir(dai.node.MonoCamera).setMaxOutputFrameSize(1600000)

# # Properties
# dir(dai.MonoCameraProperties.SensorResolution)

# monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoLeft.setCamera("left")
# monoLeft.setFps(30)
# monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoRight.setCamera("right")
# monoRight.setFps(30)
# # Create a colormap

# colormap = pipeline.create(dai.node.ImageManip)
# colormap.initialConfig.setColormap(dai.Colormap.STEREO_TURBO, stereo.initialConfig.getMaxDisparity())
# colormap.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
# colormap.setMaxOutputFrameSize(3000000)





# # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
# stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
# stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
# stereo.setLeftRightCheck(lr_check)
# stereo.setExtendedDisparity(extended_disparity)
# stereo.setSubpixel(subpixel)

# # Linking ## monoleft's output will send (link) to depth camera's left cam
# monoLeft.out.link(stereo.left)
# monoRight.out.link(stereo.right)

# stereo.disparity.link(colormap.inputImage)
# ##depth.disparity.link(xout.input)
# stereo.depth.link(xoutDepth.input)
# colormap.out.link(xoutDisparity.input)
# ct = 0
# # Connect to device and start pipeline
# with dai.Device(pipeline) as device:


#     calibData = device.readCalibration()
#     intrinsics_l = calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT) #LEFT is deprecated, use CAM_B or address camera by name  instead.
#     intrinsics_r = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT) #RIGHT is deprecated, use CAM_C or address camera by name  instead.
#     print('leftt mono camera focal length in pixels:', intrinsics_l[0][0])
#     print('Right mono camera focal length in pixels:', intrinsics_r[0][0])

#     # Output queue will be used to get the disparity frames from the outputs defined above
#     q = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
#     depthQ = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
#     # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
#     q_rgb = device.getOutputQueue("rgb")




#     while True:
#         inDisparity = q.get()  # blocking call, will wait until a new data has arrived
#         inDepth = depthQ.get()
#         in_rgb = q_rgb.tryGet()
#         print(ct)
#         ct +=1
#         if in_rgb is not None:
#             # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
#             rgb_frame = in_rgb.getCvFrame()
#             #cv2.imshow('rgb_frame', rgb_frame)
#             cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{time.time()}.jpg", rgb_frame)


#         if inDisparity is not None:
#             disparityFrame = inDisparity.getCvFrame()
#             #cv2.imshow('disparityFrame', disparityFrame)
#             cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparityFrame_{time.time()}.jpg", disparityFrame)
        
#         if inDepth is not None:
#             depthFrame = inDepth.getCvFrame().astype(np.uint16)
#             ##NOne determinded/invalid values are set to 0
#             cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{time.time()}.jpg", depthFrame)
#             #with open("depth.pickle", 'wb') as f:
#             #    pickle.dump(depthFrame, f)

#         #disparityFrame = getDisparityFrame(depthFrame, cvColorMap)

#         print(depthFrame)
#         #disparityFrame = inDisparity.getCvFrame()
        
        
#         print('rgb_frame.shape', rgb_frame.shape)
#         print('depthFrame.shape', disparityFrame.shape)
#         print('depthFrame.shape', depthFrame.shape) ##depthFrame has values of depth.
#         time.sleep(1)
#         print(f'running..{time.time()}')

#         #cv2.imshow("rgb", rgb_frame) ## disparityFrame shows rgb values better in imshow.
#         #cv2.imshow("disparity", disparityFrame)
#         ##cv2.imshow("disparity", depthFrame)

#         #depth = intrinsics_l[0][0] * 7.5 / disparityFrame
#         #break
#         #disparityFrame = (disparityFrame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

#         #cv2.imshow("disparity", disparityFrame)
#         if cv2.waitKey(1) == ord('q'):
#             break

#         '''
#         MaskDINO inference

#         MaskDINO mask * depth data

#         count number of pixel

#         divided by num of pixel

#         this is the final result.

#         '''

# """
#     while True:
#         inDisparity = q.get()  # blocking call, will wait until a new data has arrived
#         frame = inDisparity.getFrame()
#         # Normalization for better visualization
#         frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

#         cv2.imshow("disparity", frame)

#         # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
#         frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
#         cv2.imshow("disparity_color", frame)

#         if cv2.waitKey(1) == ord('q'):
#             break"""











# ##########################
# ## With RGB images, inference container and return bounding box of the container













# ##############################
# ## get all the depth values within the bounding box of the container
# ## and average out the value









# ############################







# """

# def hard_exceptions(height):
    
#     ##각 자동차 제원을 확인하여, 확실하게 높고 확실하게 낮은 높이의 무엇인가가 있으면 여기서 하드코딩으로 빼주는 작업을 하자.

#     ##
#     if height > 290:
#         raise ValueError('사진이 화물차 머리 부분이나 그에 상응하는 부분도 촬영하고 있으므로 에러를 발생합니다.')
    
#     ##화물차의 화물 컨테이너 높이 부분에 맞다면,
#     elif height <= 270 & height >= 245:
        
#         ## do segmentation
#         ## 세그멘테이션을 했을때 화물 컨테이너부분과 height의 교집합이 우리가 원하는 실제 컨테이너 부분이 될거고, 두 부분이 맞지 않는 부분이 있다면 픽셀의 갯수를 세어서 
#         ## 맞지 않는 정도를 나타내 주자.

#         ## do gradnule analysis and depth analysis 

#         ## https://docs.luxonis.com/software/depthai-components/nodes/stereo_depth/
#         ## https://docs.luxonis.com/software/depthai-components/messages/spatial_img_detections/
#         ## https://docs.luxonis.com/software/depthai-components/nodes/mobilenet_spatial_detection_network/
        
#         ## x,y,z coordinates를 안다면, 수직으로 내려오는 뎁스카메라의 높이가 아니어도 유클리디안 거리를 사용하여 실제 거리를 측정 할 수 있다. 

#     ##
#     elif height < 150: 


    


# ##아니면 먼저 트리거 이후 MaskDINO를 사용하고, 그 depth가 우리가 원하는 레인지 안에 들어오는 것만 유효하게 가져와서 부피진행한다.

# def hard_exceptions(height):
    
#     ##각 자동차 제원을 확인하여, 확실하게 높고 확실하게 낮은 높이의 무엇인가가 있으면 여기서 하드코딩으로 빼주는 작업을 하자.

#     ##
#     if height > 290:
#         raise ValueError('사진이 화물차 머리 부분이나 그에 상응하는 부분도 촬영하고 있으므로 에러를 발생합니다.')
    
#     ##화물차의 화물 컨테이너 높이 부분에 맞다면,
#     elif height <= 270 & height >= 245:
        
#         ## do segmentation

#         ## do gradnule analysis and depth analysis 


#     ##
#     elif height < 150: 

# """