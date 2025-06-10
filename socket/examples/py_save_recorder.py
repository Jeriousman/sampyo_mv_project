
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

from arena_api.__future__.save import Recorder
from arena_api.buffer import BufferFactory
from arena_api.enums import PixelFormat
from arena_api.system import system

"""
Save Recorder: Introduction
	This example demonstrates creating a save recorder
	to save videos from image buffer data. This includes,
	configuring and initializing save recorder, getting and
	appending buffers to the recorder, saving the video.
"""
TAB1 = "  "
TAB2 = "    "

def create_devices_with_tries():
	
	'''
	A function that lets example users know that a device is needed and that gives
	them a chance to connected a device instead of rasing an exception
	'''
	tries = 0
	tries_max = 5
	sleep_time_secs = 10
	while tries < tries_max:  # waits for devices for a min
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
		print(f'{TAB1}No device found! Please connect a device and run the '
			f'example again.')
		return


def example_entry_point():
	"""
	demonstrates Save Recorder
	 (1) Setup stream nodes
	 (2) Configure device nodes
	 (3) Create a recorder object and configure its width, height,
	 	and acquisition frame rate
	 (4) Open recorder, start stream and get buffers
	 (5) Append buffers to the recorder
	 (6) Close the recorder to save the video
	"""

	# Get connected devices ---------------------------------------------------

	"""
	create_device function with no arguments would create a list of
	device objects from all connected devices
	"""
	devices = create_devices_with_tries()
	if not devices:
		return

	device = system.select_device(devices)
	nodemap = device.nodemap
	tl_stream_nodemap = device.tl_stream_nodemap

	# Enable stream auto negotiate packet size
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

	# Enable stream packet resend
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	# Set node values ---------------------------------------------------------

	"""
	set width and height to max values might make the video frame rate low
	The larger the height of the buffer the lower the fps
	"""
	width_node = nodemap['Width']
	width = nodemap['Width'].max

	height_node = nodemap['Height']
	height = nodemap['Height'].max

	"""
	if the images from the device are already in the format expected by the
	recorder then no need to convert received buffers which results in better
	performance
	"""
	nodemap['PixelFormat'].value = PixelFormat.BGR8

	# start stream ------------------------------------------------------------

	with device.start_stream(100):
		print(f'{TAB1}Stream started')

		"""
		create a recorder
		The recorder, takes width, height, and frames per seconds.
		These argument can be deferred until Recorder.open is called
		"""
		recorder = Recorder(nodemap['Width'].value,
							nodemap['Height'].value,
							nodemap['AcquisitionFrameRate'].value)
		print(f"{TAB2}fps {nodemap['AcquisitionFrameRate'].value}")

		recorder.codec = ('h264', 'mp4', 'bgr8')  # order does not matter
		recorder.pattern = 'videos/py_save_recorder/video_<count>.mp4'

		"""
		recorder settings can not be changed after open is called util
		close is called
		"""
		recorder.open()
		print(f'{TAB1}Recorder opened')

		TOTAL_IMAGES = 100
		for count in range(TOTAL_IMAGES):
			buffer = device.get_buffer()
			print(f'{TAB1}Image buffer received')

			"""
			After recorder.open() add image to the open recorder stream by
				appending buffers to the video.
			The buffers are already BGR8, because we set 'PixelFormat'
				node to 'BGR8', so no need to convert buffers using
				BufferFactory.convert() from arena_api.buffer
			"""

			"""
			default name for the video is 'video<count>.mp4' where count
			is a pre-defined tag that gets updated every time open()
			is called. More custom tags can be added using
			Recorder.register_tag() function
			"""
			recorder.append(buffer)
			print(f'{TAB1}Image buffer {count} appended to video')

			device.requeue_buffer(buffer)
			print(f'{TAB1}Image buffer requeued')

		recorder.close()
		print(f'{TAB1}Recorder closed')
		print(f'{TAB1}Video saved {recorder.saved_videos[-1]}')

		video_length_in_secs = (TOTAL_IMAGES /
								nodemap['AcquisitionFrameRate'].value)
		print(f'{TAB1}Video length is {video_length_in_secs} seconds')

	# clean up ----------------------------------------------------------------

	"""
	This function call with no arguments will destroy all of the
	created devices. Having this call here is optional, if it is not
	here it will be called automatically when the system module is unloading.
	"""
	system.destroy_device()
	print(f'{TAB1}Destroyed all created devices')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!\n')
	print('Example started\n')
	example_entry_point()
	print('\nExample finished successfully')
