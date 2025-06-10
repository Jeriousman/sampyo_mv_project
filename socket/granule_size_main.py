
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
##sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from MaskDINO import train_net




# #DEPTHAI_WATCHDOG_INITIAL_DELAY=60000 DEPTHAI_BOOTUP_TIMEOUT=60000



# ################
# ## setting parameters



# ##Camera intrinsic settings
# ##https://docs.luxonis.com/software/depthai/examples/calibration_reader/
# with dai.Device() as device:
#   calibData = device.readCalibration()
#   RIGHT_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT) ##CAM_C
#   RGB_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB) ##CAM_A
#   LEFT_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT) ##CAM_B
#   print("RGB intrinsics: ", RGB_intrinsics)
#   print("RGB intrinsics type: ", type(RGB_intrinsics))
# #   print('Right mono camera focal length in pixels:', intrinsics[0][0])


# ##RGB cam intrinsic
# #[[3070.4697265625, 0.0,               1884.3509521484375], 
# # [0.0,             3069.542236328125, 1112.5499267578125], 
# # [0.0,             0.0,               1.0]]




# ## Stereo depth confidence threshold
# ## https://docs.luxonis.com/hardware/platform/depth/configuring-stereo-depth/#Configuring%20Stereo%20Depth-3.%20Improving%20depth%20accuracy-Stereo%20Subpixel%20mode
# pipeline = dai.Pipeline()
# disparity_confidence_threshold=230 ##0 is the best confidence, 255 is the worst confidence
# # Create the StereoDepth node
# stereo_depth = pipeline.create(dai.node.StereoDepth)
# stereo_depth.initialConfig.setConfidenceThreshold(disparity_confidence_threshold)

# # Or, alternatively, set the Stereo Preset Mode:
# # Prioritize fill-rate, sets Confidence threshold to 245
# stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# # Prioritize accuracy, sets Confidence threshold to 200
# stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)






















#######################
## Connect Depth Camera and get RGB & depth images
## https://docs.luxonis.com/software/depthai/examples/stereo_depth_video/
import cv2
import depthai as dai
import numpy as np
import pickle
from datetime import timedelta
from datetime import datetime
# def getDisparityFrame(frame, cvColorMap):
#     maxDisp = stereo.initialConfig.getMaxDisparity()
#     disp = (frame * (255.0 / maxDisp)).astype(np.uint8)
#     disp = cv2.applyColorMap(disp, cvColorMap)

#     return disp
# cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
# cvColorMap[0] = [0, 0, 0]





# Create pipeline
pipeline = dai.Pipeline()
# device = dai.Device()



# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
camRgb = pipeline.create(dai.node.ColorCamera)
##camRgb = pipeline.create(dai.node.Camera)  ##these below are for rgb
##final out links
stereo = pipeline.create(dai.node.StereoDepth) ##This catches disparity and depth.
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDisparity = pipeline.create(dai.node.XLinkOut) ##XLink sends to the host (computer)
# sync = pipeline.create(dai.node.Sync)
# sync.setSyncThreshold(timedelta(milliseconds=20))


##setting names to use in pipeline
xoutRgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")
xoutDisparity.setStreamName("disparity")
# sync_out = pipeline.createXLinkOut()
# sync_out.setStreamName("rgbd")

# # Properties
fps = 20

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A) ##RGB
camRgb.setFps(fps)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P) ## THE_400_P
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B) ##LEFT
monoLeft.setFps(fps)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P) ## THE_400_P
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C) ##RIGHT
monoRight.setFps(fps)

lr_check=True
extended_disparity=False
subpixel=True

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# stereo.setOutputSize(1248, 936)
stereo.setOutputSize(3840, 2160)
# stereo.setOutputSize(4048, 3040)
stereo.setLeftRightCheck(lr_check) # Better handling for occlusions:
stereo.setExtendedDisparity(extended_disparity) # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
stereo.setSubpixel(subpixel) # Better accuracy for longer distance, fractional disparity 32-levels:
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)


# Linking
camRgb.video.link(xoutRgb.input)  ##camRgb를 xoutRgb의 인풋으로 보낸다는 뜻.
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
stereo.disparity.link(xoutDisparity.input)
## When using Sync, use this linking

# monoLeft.out.link(stereo.left)
# monoRight.out.link(stereo.right)
# stereo.depth.link(sync.inputs["depth"])
# camRgb.video.link(sync.inputs["rgb"])  ##camRgb를 sync의 인풋으로 보낸다는 뜻.
# sync.out.link(sync_out.input)

##stereo.disparity.link(colormap.inputImage)
##depth.disparity.link(xout.input)
##colormap.out.link(xoutDisparity.input)





# ct = 0
# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise

    calibData = device.readCalibration()
    intrinsics_l = calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT) #LEFT is deprecated, use CAM_B or address camera by name  instead.
    intrinsics_r = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT) #RIGHT is deprecated, use CAM_C or address camera by name  instead.
    print('leftt mono camera focal length in pixels:', intrinsics_l[0][0])
    print('Right mono camera focal length in pixels:', intrinsics_r[0][0])
    print('Right mono camera focal length in pixels:', intrinsics_r[0][0])
    # Output queue will be used to get the disparity frames from the outputs defined above
    q_disparity = device.getOutputQueue(name="disparity")##, maxSize=1, blocking=False)
    q_depth = device.getOutputQueue(name="depth")
    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")


    while True:

        in_disparity = q_disparity.get()  # blocking call, will wait until a new data has arrived
        in_depth = q_depth.get()
        in_rgb = q_rgb.tryGet()


        # print(ct)
        # ct +=1
        currentDateAndTime = datetime.now()
        if in_rgb is not None:
            
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            rgb_frame = in_rgb.getCvFrame()
            cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}.jpg", rgb_frame)

        if in_disparity is not None:
            disparity_frame = in_disparity.getCvFrame()
            cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparity_frame_{currentDateAndTime}.jpg", disparity_frame)

        if in_depth is not None:
            depth_frame = in_depth.getCvFrame()##.astype(np.uint16)
            ##NOne determinded/invalid values are set to 0
            cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}.jpg", depth_frame)
            with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_{currentDateAndTime}.pickle", 'wb') as f:
               pickle.dump(depth_frame, f)
        
        #disparityFrame = getDisparityFrame(depthFrame, cvColorMap)

        # print(depthFrame)
        #disparityFrame = inDisparity.getCvFrame()
        
        
        print('rgb_frame.shape', rgb_frame.shape)
        print('disparity_frame.shape', disparity_frame.shape)
        print('depth_frame.shape', depth_frame.shape) ##depthFrame has values of depth.

        # cv2.imshow("rgb", rgb_frame) ## disparityFrame shows rgb values better in imshow.
        #cv2.imshow("disparity", disparityFrame)
        ##cv2.imshow("disparity", depthFrame)



# with device:
#     device.startPipeline(pipeline)

#     while True:
#         messages = device.getOutputQueue("rgbd").tryGetAll()
#         if not messages:
#             continue
#         for message_group in messages:
#             for name, frame in message_group:
#                 print(f"{name}: {frame.getSequenceNum()}")
#             print("================= End MessageGroup ================")





        #cv2.imshow("disparity", disparityFrame)
        if cv2.waitKey(1) == ord('q'):
            break



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



# with open("/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_2024-12-26 14:28:48.650297.pickle", 'rb') as f:
#     depth_array = pickle.load(f)

# depth_array

# # cam_height = np.zeros(shape=(2160, 3840))
# cam_height = np.full((2160, 3840), 12000)
# cam_height - depth_array
# max(depth_array)
# np.where(depth_array>10000).
# np.count_nonzero(depth_array < 10000)
# np.sum(depth_array < 10000)
# 6023143 + 2271257
# 2160 * 3840
#np.mean(depth_array)

# map(max, depth_array)
# largest = list(map(max, depth_array))
# smallest = list(map(min, depth_array))
        '''
        MaskDINO inference

        MaskDINO mask * depth data

        count number of pixel

        divided by num of pixel

        this is the final result.

        '''

"""
    while True:
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getFrame()
        # Normalization for better visualization
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        cv2.imshow("disparity", frame)

        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", frame)

        if cv2.waitKey(1) == ord('q'):
            break"""











##########################
## With RGB images, inference container and return bounding box of the container













##############################
## get all the depth values within the bounding box of the container
## and average out the value









############################









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

