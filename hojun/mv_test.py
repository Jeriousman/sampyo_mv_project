import cv2
import numpy as np
import datetime
from PIL import Image
#import sys
#sys.path.append('/home/sdt/Workspace/sampyo/granule-analysis/sam2')
from arena_api.system import system
print(system)
import time
# device_infos = system.device_infos
# print(device_infos)
# device = system.create_device(device_infos=device_infos[0])[0]
# device

# device.start_stream(1)
# try:
#     while True:
#        image_buffer = device.get_buffer()
#        ##ctypeslib is a module that makes C library run in python. ##pdata = pixel data?
#        ##(3000, 4096, 1) shape
#        np_img = np.ctypeslib.as_array(image_buffer.pdata,shape=(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))).reshape(image_buffer.height, image_buffer.width, int(image_buffer.bits_per_pixel / 8))
#        ##(3000, 4096, 3) shape. Grayscale to RGB 
#        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BayerRG2RGB)
#        #cv2.imshow('lucid_img', rgb_img)
#        cv2.imwrite(f'/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb/mv_frame_{time.time()}.jpg', rgb_img)
#        device.requeue_buffer(image_buffer)
#        #if cv2.waitKey(1) != -1:
#        #    cv2.destroyAllWindows()
#        #    device.stop_stream()
#        #break
#        #cv2.imwrite('./lucid_rgb.jpg', rgb_img)
#        #print('save img')
#        time.sleep(1)
#        print(f'running...{time.time()}')
#     device.stop_stream()
# except Exception as e:
#     print(e)
#     device.stop_stream()


class MVCam:
    def __init__(self):
        try:
            device_infos = system.device_infos
            print(device_infos)
            self.device = system.create_device(device_infos=device_infos[0])[0]  

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
            # logger.info(f"self.device.nodemap - ExposureAuto, ExposureTime : {nodes['ExposureAuto'].value, nodes['ExposureTime'].value}")
            
            nodes['OffsetX'].value = 0
            nodes['OffsetY'].value = 0
            # logger.info(f"self.device.nodemap - OffsetX, OffsetY : {nodes['OffsetX'].value}, {nodes['OffsetY'].value}")
        
            # to acquisition rapid, packet size setting: origin = 1500
            stream_packet_size_max = nodes['DeviceStreamChannelPacketSize'].max
            nodes['DeviceStreamChannelPacketSize'].value = stream_packet_size_max
            # logger.info(f"self.device.nodemap - DeviceStreamChannelPacketSize, packet size max : {nodes['DeviceStreamChannelPacketSize'].value}, {nodes['DeviceStreamChannelPacketSize'].max}")
        
        
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
            # logger.error(traceback.format_exc())
            #self.device.stop_stream()
            system.destroy_device()
            # logger.error(traceback.format_exc())
            # logger.error('Destroyed all created devices')
           



    def capture(self):
    # def capture(self):
        try:
            while True:
                ### added by dykim ###
                self.device.start_stream(1)
                # logger.info('Started stream')
                # logger.info('Grabbing an image buffer')
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
                # logger.info(f"MV rgb img saved at : {current_time_mv}") 

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
            # logger.info(traceback.format_exc())




hojun = MVCam()
text = hojun.capture()