

import time
from arena_api.system import system


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


def store_initial_nodes(device):
	'''
	Stores the initial values of nodes that will be modified in this example
	'''
	
	nodemap = device.nodemap
	nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat', 'ExposureAuto',
							'ExposureTime', 'DeviceStreamChannelPacketSize'])
	width_initial = nodes['Width'].value
	height_initial = nodes['Height'].value
	pixel_format_initial = nodes['PixelFormat'].value
	exposure_auto_initial = nodes['ExposureAuto'].value
	exposure_time_initial = nodes['ExposureTime'].value
	stream_packet_size_initial = nodes['DeviceStreamChannelPacketSize'].value

	initial_values = [width_initial, height_initial, pixel_format_initial,
					exposure_time_initial, exposure_auto_initial,
					stream_packet_size_initial]
	return nodes, initial_values


def setup(device, nodes, width, height, pixel_format, exposure_auto):
	'''
	Set features before streaming
	'''
	if nodes['Width'].is_readable and nodes['Width'].is_writable:
		nodes['Width'].value = width
	if nodes['Height'].is_readable and nodes['Height'].is_writable:
		nodes['Height'].value = height
	if nodes['PixelFormat'].is_readable and nodes['PixelFormat'].is_writable:
		nodes['PixelFormat'].value = pixel_format
	if nodes['ExposureAuto'].is_readable and nodes['ExposureAuto'].is_writable:
		nodes['ExposureAuto'].value = exposure_auto
	if nodes['ExposureTime'].is_readable and nodes['ExposureTime'].is_writable:
		nodes['ExposureTime'].value = nodes['ExposureTime'].min
		# nodes['ExposureTime'].value = 1000.0
	if nodes['DeviceStreamChannelPacketSize'].is_readable \
	and nodes['DeviceStreamChannelPacketSize'].is_writable:
		'''
		Set maximum stream channel packet size
			Maximizing packet size increases frame rate by reducing the amount of
			overhead required between images. This includes both extra
			header/trailer data per packet as well as extra time from
			intra-packet spacing (the time between packets). In order to grab
			images at the maximum packet size, the Ethernet adapter must be
			configured appropriately: 'Jumbo packet' must be set to its maximum,
			'UDP checksum offload' must be set to 'Rx & Tx Enabled', and
			'Received Buffers' must be set to its maximum.
		'''
		stream_packet_size_max = nodes['DeviceStreamChannelPacketSize'].max
		# stream_packet_size_max = 1
		nodes['DeviceStreamChannelPacketSize'].value = stream_packet_size_max

	else:
		raise Exception(f'Read/Write access to devices nodes not available. '
						f'Reconnecting device recommended')

	stream_nodemap = device.tl_stream_nodemap
	stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	stream_nodemap['StreamPacketResendEnable'].value = True


def return_original(nodes, initial_values):
	'''
	Returns the nodes to their initial value
	'''
	nodes['Width'].value = initial_values[0]
	nodes['Height'].value = initial_values[1]
	nodes['PixelFormat'].value = initial_values[2]
	nodes['ExposureTime'].value = initial_values[3]
	nodes['ExposureAuto'].value = initial_values[4]
	nodes['DeviceStreamChannelPacketSize'].value = initial_values[5]


def acquire_images_rapidly(device, nodes):
	'''
	demonstrates configuration for high frame rates
	(1) lowers image size
	(2) maximizes packet size
	(3) minimizes exposure time
	(4) sets large number of buffers
	(5) waits until after acquisition to requeue buffers
	'''

	'''
	Set low width and height
		Reducing the size of an image reduces the amount of bandwidth required
		for each image. The less bandwidth required per image, the more images
		can be sent over the same bandwidth.
	'''
	width = 2700 ##4096
	height = 2000 ##3000
	print(f'{TAB1}Set low width and height ({width}x{height})')

	'''
	Set small pixel format
		Similar to reducing the ROI, reducing the number of bits per pixel also
		reduces the bandwidth required for each image. The smallest pixel formats
		are 8-bit bayer and 8-bit mono (i.e. BayerRG8 and Mono8).
	'''
	pixel_format = 'Mono8' ## 'Mono8'  ##'BayerRG8'
	print(f'{TAB1}Set small pixel format ({pixel_format})')

	'''
	Set low exposure time 
		Decreasing exposure time can increase frame rate by reducing the amount
		of time it takes to grab an image. Reducing the exposure time past
		certain thresholds can cause problems related to not having enough light.
		This can sometimes be mitigated by increasing gain and/or
		environmental light.
	'''
	exposure_auto = "Once" ##"Off" ##'Continuous' ##27362.104 ##'Off' ##27362.104

	setup(device, nodes, width, height, pixel_format, exposure_auto)

	exposure_min = nodes['ExposureTime'].value
	print(f'{TAB1}Set minimum exposure time ({exposure_min})')

	'''
	Start stream with large number of buffers
		Increasing the number of buffers can increase speeds by reducing the
		amount of time taken to requeue buffers. In this example, one buffer is
		used for each image. Of course, the number of buffers that can be used is
		limited by the amount of space in memory.
	'''
	num_buffers = 1

	'''
	Starting the stream allocates buffers, which can be passed in as
		an argument (default: 10), and begins filling them with data. Buffers
		must later be requeued to avoid memory leaks.
	'''
	
	with device.start_stream(num_buffers):
		print(f'{TAB1}Stream started with {num_buffers} buffers')
		""" 'device.get_buffer(arg)' returns arg number of buffers
        the buffer is in the rgb layout """
		
		print(f'{TAB1}Get {num_buffers} buffers')

        # Grab images --------------------------------------------------------
		buffers = device.get_buffer(num_buffers)

        # Print image buffer info
		for count, buffer in enumerate(buffers):
			print(f'{TAB2}buffer{count:{2}} received | '
                  f'Width = {buffer.width} pxl, '
                  f'Height = {buffer.height} pxl, '
                  f'Pixel Format = {buffer.pixel_format.name}')
				  
		device.requeue_buffer(buffers)
		print(f'{TAB1}Requeued {num_buffers} buffers')

def acquire_images_setting(device, nodes):
	'''
	demonstrates configuration for high frame rates
	(1) lowers image size
	(2) maximizes packet size
	(3) minimizes exposure time
	(4) sets large number of buffers
	(5) waits until after acquisition to requeue buffers
	'''

	'''
	Set low width and height
		Reducing the size of an image reduces the amount of bandwidth required
		for each image. The less bandwidth required per image, the more images
		can be sent over the same bandwidth.
	'''
	width = 2700 ##2700 ##4096
	height = 2000 ##2000 ##3000
	print(f'{TAB1}Set low width and height ({width}x{height})')

	'''
	Set small pixel format
		Similar to reducing the ROI, reducing the number of bits per pixel also
		reduces the bandwidth required for each image. The smallest pixel formats
		are 8-bit bayer and 8-bit mono (i.e. BayerRG8 and Mono8).
	'''
	pixel_format = 'Mono8' ##'BayerRG8'##'Mono8'  ##'BayerRG8'
	print(f'{TAB1}Set small pixel format ({pixel_format})')

	'''
	Set low exposure time 
		Decreasing exposure time can increase frame rate by reducing the amount
		of time it takes to grab an image. Reducing the exposure time past
		certain thresholds can cause problems related to not having enough light.
		This can sometimes be mitigated by increasing gain and/or
		environmental light.
	'''
	exposure_auto = "Continuous"##"Once"'Continuous' ##"Off" ##'Continuous' ##27362.104 ##'Off' ##27362.104

	setup(device, nodes, width, height, pixel_format, exposure_auto)

	exposure_min = nodes['ExposureTime'].value
	print(f'{TAB1}Set minimum exposure time ({exposure_min})')

	'''
	Start stream with large number of buffers
		Increasing the number of buffers can increase speeds by reducing the
		amount of time taken to requeue buffers. In this example, one buffer is
		used for each image. Of course, the number of buffers that can be used is
		limited by the amount of space in memory.
	'''
	# num_buffers = 1

	'''
	Starting the stream allocates buffers, which can be passed in as
		an argument (default: 10), and begins filling them with data. Buffers
		must later be requeued to avoid memory leaks.
	'''
	
	# device.start_stream(num_buffers)


	
def example_entry_point():

    # Get devices
    devices = create_devices_with_tries()
    device = system.select_device(devices)

	device_infos = system.device_infos
	device = system.create_device(device_infos=self.device_infos[0])[0]
    # Store initial values
    nodes, initial_values = store_initial_nodes(device)
	
    # Acquire images
    acquire_images_setting(device, nodes)
    # acquire_images_rapidly(device, nodes)
    nodes, values = store_initial_nodes(device)
    print(values)
    system.destroy_device() ##very important

	# system.device_infos

	# nodemap = device.nodemap
	# nodes = nodemap.get_node(['PacketResendWindowFrameCount',
	# 						'DeviceLinkThroughputReserve',
	# 						'Width',
	# 						'Height',
	# 						'PixelFormat',
	# 						'ExposureAuto',
	# 						'ExposureTime',
	# 						'DeviceStreamChannelPacketSize'])
	# nodemap["PacketResendWindowFrameCount"] ##Original value 4
	# nodemap["DeviceLinkThroughputReserve"] ##Original value 10
	# nodemap['Width'] ##Original value 4096
	# nodemap['Height'] ##Original value 3000
	# nodemap['PixelFormat'] ##Original value BayerRG8
	# nodemap['ExposureAuto'] ##Original value Continuous
	# nodemap['ExposureTime'] ##Original value 110076.576
	# nodemap['DeviceStreamChannelPacketSize'] ##Original value 9000
	# # nodemap['StreamAutoNegotiatePacketSize'] ##Original value 9000
	# stream_nodemap = device.tl_stream_nodemap
	# stream_nodemap['StreamAutoNegotiatePacketSize']##Original value False
	# stream_nodemap['StreamPacketResendEnable']##Original value False

	# nodemap["PacketResendWindowFrameCount"].value=4 ##Original value 4
	# nodemap["DeviceLinkThroughputReserve"].value=12 ##Original value 10
	# nodemap['Width'].value=2700 ##Original value 4096
	# nodemap['Height'].value=2000 ##Original value 3000
	# nodemap['PixelFormat'].value="Mono8" ##Original value BayerRG8
	# nodemap['ExposureAuto'].value ##Original value Continuous
	# nodemap['ExposureTime'].value ##Original value 110076.576
	# nodemap['DeviceStreamChannelPacketSize'].value=9000 ##Original value 9000
	# # nodemap['StreamAutoNegotiatePacketSize'].value ##Original value 9000
	# stream_nodemap = device.tl_stream_nodemap
	# stream_nodemap['StreamAutoNegotiatePacketSize'].value=True##Original value False
	# stream_nodemap['StreamPacketResendEnable'].value=True##Original value False


    # # Acquire images
	# acquire_images_setting(device, nodes)
    # # acquire_images_rapidly(device, nodes)
	# nodes, initial_values = store_initial_nodes(device)
	# system.destroy_device() ##very important
	# system.device_infos
    # # Return nodes to original value
    # print(f'{TAB1}Returning nodes to their initial value')
    # return_original(nodes, initial_values)

    # # Stop stream
    # device.stop_stream()
    # print(f'{TAB1}Stream stopped')

    # # Destroy device
    # system.destroy_device()
    # print(f'{TAB1}Destroyed all created devices')


if __name__ == '__main__':
#     print('Example started\n')
    example_entry_point()
#     print('\nExample finished successfully')
# ##[2700, 2000, 'Mono8', 50745.64, 'Continuous', 1500]  2025-01-14
# dir(device)
# device._Device__WAIT_ON_EVENT_TIMEOUT_MILLISEC
# device._Device__GET_BUFFER_TIMEOUT_MILLISEC
# device._Device__get_GET_BUFFER_TIMEOUT_MILLISEC()
# device._Device__get_WAIT_ON_EVENT_TIMEOUT_MILLISEC()
# device._Device__set_GET_BUFFER_TIMEOUT_MILLISEC()
# device._Device__set_WAIT_ON_EVENT_TIMEOUT_MILLISEC()
# device._Device__throw_if_get_buffer_is_called_before_start_stream()
# device.initialize_events()
# device.is_connected()
# device._Device__DEFAULT_NUM_BUFFERS


# device.start_stream()
# 	buffers = device.get_buffer()
# print(f'{TAB1}Stream started with {num_buffers} buffers')
# 	""" 'device.get_buffer(arg)' returns arg number of buffers
# 	the buffer is in the rgb layout """
	
# 	print(f'{TAB1}Get {num_buffers} buffers')

# 	# Grab images --------------------------------------------------------
# 	buffers = device.get_buffer(num_buffers)

# 	# Print image buffer info
# 	for count, buffer in enumerate(buffers):
# 		print(f'{TAB2}buffer{count:{2}} received | '
# 				f'Width = {buffer.width} pxl, '
# 				f'Height = {buffer.height} pxl, '
# 				f'Pixel Format = {buffer.pixel_format.name}')