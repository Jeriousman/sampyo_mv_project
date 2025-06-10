# -----------------------------------------------------------------------------
# Copyright (c) 2024, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

import time
from datetime import datetime
from arena_api import enums as _enums
from arena_api.enums import PixelFormat
from arena_api.__future__.save import Writer
from arena_api.system import system
from arena_api.buffer import BufferFactory

'''
Image Processing: Split Channels
   This example introduces splitting channels of an image and saving split images as JPEG images.
'''

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''
TAB1 = "  "
TAB2 = "    "
pixel_format = PixelFormat.BGR8


def create_device_with_tries():
	'''
	Waits for the user to connect a device before raising
		an exception if it fails
	'''
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	devices = None
	while tries < tries_max:
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
			return devices
	else:
		raise Exception(f'{TAB1}No device found! Please connect a device and run '
						f'the example again.')


def save(buffer, count):
	'''
	demonstrates saving a JPEG image
	(1) converts image to a displayable pixel format
	(2) prepares image parameters
	(3) prepares image writer
	(4) saves image with configuration parameters
	(5) destroys converted image
	'''

	'''
	convert image
	'''
	converted = BufferFactory.convert(buffer, pixel_format)
	print(f"{TAB1}Converted image to {pixel_format.name}")

	'''
	prepare image writer
	'''
	print(f'{TAB1}Prepare Image Writer')
	writer = Writer()
	writer.pattern = 'images/py_imageprocessing_splitchannels/image_'+ str(count)+'.jpg'
	
	writer.save(converted, quality=75, progressive=False, subsampling=_enums.ScJpegSubsamplingList.SC_NO_JPEG_SUBSAMPLING, optimize=False)
	print(f'{TAB1}Image saved {writer.saved_images[-1]}')

	# Destroy converted buffer to avoid memory leaks
	BufferFactory.destroy(converted)


def split_and_save(buffer):

	# Split channels

	print(f'{TAB1}Split Channels')

	buffers = BufferFactory.split_channels(buffer)

	for count, buf in enumerate(buffers):
		save(buf, count)


def example_entry_point():
	devices = create_device_with_tries()
	device = system.select_device(devices)

	'''
	Setup stream values
	'''
	tl_stream_nodemap = device.tl_stream_nodemap
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	device.start_stream()

	buffer = device.get_buffer()

	split_and_save(buffer)

	device.requeue_buffer(buffer)

	# Clean up
	device.stop_stream()

	# Destroy Device
	system.destroy_device()
	
	print(f'{TAB1}Destroyed all created devices')


if __name__ == "__main__":
	print("Example Started\n")
	example_entry_point()
	print("\nExample Completed")
