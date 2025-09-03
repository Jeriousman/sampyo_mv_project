#!/usr/bin/env python3
import depthai as dai
from datetime import timedelta
from datetime import datetime
import cv2
import time
import pickle
###########################################################################################
## uncomment this for depth+RGB blending
# Weights to use when blending depth/rgb image (should equal 1.0)
# rgbWeight = 0.4
# depthWeight = 0.6
# def updateBlendWeights(percent_rgb):
#     """
#     Update the rgb and depth weights used to blend depth/rgb image
#     @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
#     """
#     global depthWeight
#     global rgbWeight
#     rgbWeight = float(percent_rgb)/100.0
    # depthWeight = 1.0 - rgbWeight
###########################################################################################




# ##########################################################################################
# ##Camera intrinsic settings
# ##https://docs.luxonis.com/software/depthai/examples/calibration_reader/
# with dai.Device() as device:
#   calibData = device.readCalibration()
#   RIGHT_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C) ##CAM_C
#   RGB_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A) ##CAM_A
#   LEFT_intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B) ##CAM_B
#   print("RGB intrinsics: ", RGB_intrinsics)
#   print("RGB intrinsics type: ", type(RGB_intrinsics))
# #   print('Right mono camera focal length in pixels:', intrinsics[0][0])


# ##RGB cam intrinsic
# #[[3070.4697265625, 0.0,               1884.3509521484375], 
# # [0.0,             3069.542236328125, 1112.5499267578125], 
# # [0.0,             0.0,               1.0]]
# ##########################################################################################




class depth_cam:

    def __init__(self):


        self.fps = 20
        # The disparity is computed at this resolution, then upscaled to RGB resolution
        #monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        self.monoResolution = dai.MonoCameraProperties.SensorResolution.THE_800_P

        # Create pipeline
        self.pipeline = dai.Pipeline()
        self.device = dai.Device()

        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.left = self.pipeline.create(dai.node.MonoCamera)
        self.right = self.pipeline.create(dai.node.MonoCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)

        # rgbOut = pipeline.create(dai.node.XLinkOut)
        # rgbOut.setStreamName("rgb")


        try:
            calibData = self.device.readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lensPosition:
                self.camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise

        #Properties
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        #camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP) # 4056x3040
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
        self.h, self.w = 1248*2, 936*2
        # # 4056x3040
        self.stereo.setOutputSize(self.h, self.w)
        #stereo.setOutputSize(1280, 720)
        # stereo.setOutputSize(4056, 3040)
        self.sync = self.pipeline.create(dai.node.Sync)
        self.sync.setSyncThreshold(timedelta(milliseconds=30))

        # Linking
        self.camRgb.isp.link(self.sync.inputs["rgb"])
        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)
        self.stereo.depth.link(self.sync.inputs["depth"])


        self.sync_out = self.pipeline.createXLinkOut()
        self.sync_out.setStreamName("rgbd")
        self.sync.out.link(self.sync_out.input)



            # while True:
            #     pass
# while True:
    def capture(self):
        # Connect to device and start pipeline
        with self.device:
            self.device.startPipeline(self.pipeline)
            while True:
                self.messages = self.device.getOutputQueue("rgbd").tryGetAll()    
                # while True:
                start = time.time() 
                # messages = self.device.getOutputQueue("rgbd").tryGetAll()
                if not self.messages:
                    continue
                currentDateAndTime = datetime.now()
                
                for message_group in self.messages:
                    for name, frame in message_group:
                        print(f"{name}: {frame.getSequenceNum()}")

                        if name == 'depth':
                            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                            depth_frame = frame.getCvFrame()##.astype(np.uint16)
                            cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}.jpg", depth_frame)
                            print('depth iamge collected')
                            # with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_{currentDateAndTime}.pickle", 'wb') as f:
                            #     pickle.dump(depth_frame, f)
                        ###########################################################################################
                        ## uncomment this for depth+RGB blending
                            # depth_frame = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                            # depth_frame = cv2.equalizeHist(depth_frame)
                            # depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)
                        ###########################################################################################
                            # print('depth: ', depth_frame)
                            print('depth: ', depth_frame.shape)


                        # elif name == 'disparity':
                        #     disparity_frame = frame.getCvFrame()
                        #     cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparity_frame_{currentDateAndTime}.jpg", disparity_frame)


                        elif name == 'rgb':
                            rgb_frame = frame.getCvFrame()##.astype(np.uint16)
                            rgb_frame = cv2.resize(rgb_frame, (self.h, self.w), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}.jpg", rgb_frame)
                            # print('RGB: ', rgb_frame)
                            print('RGB iamge collected')
                            print('RGB: ', rgb_frame.shape)

                        end = time.time()
                        print(end - start)

                        ###########################################################################################
                        ## uncomment this for depth+RGB blending
                        # if rgb_frame is not None and depth_frame is not None:
                        #     ct+=1
                        #     # Need to have both frames in BGR format before blending
                        #     if len(depth_frame.shape) < 3:
                        #         depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
                        #     blended = cv2.addWeighted(rgb_frame, rgbWeight, depth_frame, depthWeight, 0)
                        #     cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/blended/blended_{ct}.jpg", blended)
                        #     rgb_frame = None
                        #     depth_frame = None
                        ###########################################################################################
                        


                        # if name == 'rgb':
                        #     print('rgb baby')
                        #     # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                        #     rgb_frame = frame.getCvFrame()
                        #     cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb/rgb_frame_{currentDateAndTime}.jpg", rgb_frame)

                        # if in_disparity is not None:
                        #     disparity_frame = in_disparity.getCvFrame()
                        #     cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_disparity/disparity_frame_{currentDateAndTime}.jpg", disparity_frame)

                        # elif name == 'depth':
                        #     print('depth baby')
                        #     depth_frame = frame.getCvFrame()##.astype(np.uint16)
                        #     ##NOne determinded/invalid values are set to 0
                        #     cv2.imwrite(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth/depthFrame_{currentDateAndTime}.jpg", depth_frame)
                        #     # with open(f"/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array/depthFrameArray_{currentDateAndTime}.pickle", 'wb') as f:
                        #         # pickle.dump(depth_frame, f)
                        
                        # #disparityFrame = getDisparityFrame(depthFrame, cvColorMap)

                        # # print(depthFrame)
                        # #disparityFrame = inDisparity.getCvFrame()
                        
                        # print('rgb_frame.shape', rgb_frame.shape)
                        # # print('disparity_frame.shape', disparity_frame.shape)
                        # print('depth_frame.shape', depth_frame.shape) ##depthFrame has values of depth.


if __name__=="__main__":  
    print(__name__)  
    depth_camera = depth_cam()
    depth_camera.capture()