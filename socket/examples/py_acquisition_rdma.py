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

from arena_api.system import system

'''
Acquisition: RDMA
	This example introduces RDMA stream protocol. 
    RDMA is a reliably connected transport protocol that transfers data 
	from the camera to host memory without involving the CPU.
	It features low - latency transfers and zero - copy.
	A supported RDMA camera and capable NIC are required.
'''


def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''

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
			print(f'Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')


def configure_and_get_image_buffers(device):

	nodemap = device.nodemap
	tl_stream_nodemap = device.tl_stream_nodemap
	nodes = device.nodemap.get_node(['Width', 'Height'])

	# Store initial settings, to restore later
	width_initial = nodes['Width'].value
	height_initial = nodes['Height'].value

	# Set features before streaming.-------------------------------------------
	initial_acquisition_mode = nodemap.get_node("AcquisitionMode").value

	nodemap.get_node("AcquisitionMode").value = "Continuous"
	
	tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"

	'''
	The TransportStreamProtocol node can tell the camera to use the RDMA datastream engine. When
	    set to RDMA - Arena will switch to using the RDMA datastream engine. 
	    There is no further necessary configuration, though to achieve maximum throughput 
	    users may want to set the "DeviceLinkThroughputReserve" to 0 and 
	    also set the stream channel packet delay "GevSCPD" to 0.
	'''
	print('Set Transport Stream Protocol to RDMA\n')

	try:
		transport_stream_protocol_initial = nodemap.get_node('TransportStreamProtocol').value

		nodemap["TransportStreamProtocol"].value = "RDMA"

		# Set width and height to their max values
		print('Setting \'Width\' and \'Height\' Nodes value to their '
			'max values')
		nodes['Width'].value = nodes['Width'].max
		nodes['Height'].value = nodes['Height'].max

		number_of_buffers = 20

		device.start_stream(number_of_buffers)
		print(f'Stream started with {number_of_buffers} buffers')

		print(f'Get {number_of_buffers} buffers in a list')
		buffers = device.get_buffer(number_of_buffers)

		for count, buffer in enumerate(buffers):
			print(f'\tbuffer{count:{2}} received | '
				f'Width = {buffer.width} pxl, '
				f'Height = {buffer.height} pxl, '
				f'Pixel Format = {buffer.pixel_format.name}')

		device.requeue_buffer(buffers)
		print(f'Requeued {number_of_buffers} buffers')

		device.stop_stream()
		print(f'Stream stopped')

		# Restore initial values
		nodemap.get_node("TransportStreamProtocol").value = transport_stream_protocol_initial

	except:
		print('Connected camera does not support RDMA stream\n')

	# Restore initial values
	nodemap.get_node("AcquisitionMode").value = initial_acquisition_mode
	nodemap.get_node("Width").value = width_initial
	nodemap.get_node("Height").value = height_initial


def example_entry_point():

	# Get connected devices ---------------------------------------------------

	# Create a device
	devices = create_devices_with_tries()
	device = devices[0]
	print(f'Device used in the example:\n\t{device}')

	# Configure device and grab images ----------------------------------------

	configure_and_get_image_buffers(device)

	# Clean up ----------------------------------------------------------------

	system.destroy_device()
	print('Destroyed all created devices')


if __name__ == '__main__':
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
