import os  # os.getcwd()
import time
from pathlib import Path

import numpy as np  # pip install numpy
from PIL import Image as PIL_Image  # pip install Pillow

from arena_api.system import system

tries = 0
tries_max = 6
sleep_time_secs = 10
while tries < tries_max:  # Wait for device for 60 seconds
    devices = system.create_device()
    if not devices:
        print(
            f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
            f'secs for a device to be connected!')
        for sec_count in range(sleep_time_secs):
            time.sleep(1)
            print(f'{sec_count + 1 } seconds passed ',
                  '.' * sec_count, end='\r')
        tries += 1
    else:
        print(f'Created {len(devices)} device(s)\n')
        break
else:
    raise Exception(f'No device found! Please connect a device and run '
                    f'the example again.')

device = system.select_device(devices)
print(f'Device used in the example:\n\t{device}')



# Get device stream nodemap
tl_stream_nodemap = device.tl_stream_nodemap
# Enable stream auto negotiate packet size
tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
# Enable stream packet resend
tl_stream_nodemap['StreamPacketResendEnable'].value = True
# Get/Set nodes -----------------------------------------------------------
nodes = device.nodemap.get_node(['Width', 'Height', 'PixelFormat'])



print('Setting Width to its maximum value')
nodes['Width'].value = 2700

print('Setting Height to its maximum value')
height = nodes['Height']
height.value = 2000

pixel_format_name = 'Mono8'
print(f'Setting Pixel Format to {pixel_format_name}')
nodes['PixelFormat'].value = pixel_format_name

print('Starting stream')
device.start_stream(1)


print('Grabbing an image buffer')
image_buffer = device.get_buffer()
print(f' Width X Height = ' 
    f'{image_buffer.width} x {image_buffer.height}')


## method 1 ###
image_only_data = None
if image_buffer.has_chunkdata:
    # 8 is the number of bits in a byte
    bytes_pre_pixel = int(image_buffer.bits_per_pixel / 8)

    image_size_in_bytes = image_buffer.height * \
        image_buffer.width * bytes_pre_pixel

    image_only_data = image_buffer.data[:image_size_in_bytes]
else:
    image_only_data = image_buffer.data

nparray = np.asarray(image_only_data, dtype=np.uint8)
# Reshape array for pillow
nparray_reshaped = nparray.reshape((
    image_buffer.height,
    image_buffer.width
))


### method 2 ###
nparray_reshaped = np.ctypeslib.as_array(
    image_buffer.pdata,(image_buffer.height, image_buffer.width))


### save image ###
print('Saving image')
png_name = f'from_{pixel_format_name}_to_png_with_pil.jpg'
png_array = PIL_Image.fromarray(nparray_reshaped)
png_array.save(png_name)
print(f'Saved image path is: {Path(os.getcwd()) / png_name}')

device.requeue_buffer(image_buffer)
device.stop_stream()
system.destroy_device()
print('Destroyed all created devices')