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


from arena_api.system import system

'''
Enumeration Unicast: Introduction
   This example introduces adding unicast device. This includes opening and
   closing the system, updating and retrieving the list of devices, adding
   unicast devices using the IP address for the device, and chekcing connections
   of the devices
'''

"""
 =-=-=-=-=-=-=-=-=-
 =-=- EXAMPLE -=-=-
 =-=-=-=-=-=-=-=-=-
"""
TAB1 = "  "
TAB2 = "    "

def enumerate_devices():
	'''
	enumerates device(s)
	(1) gets device list
	(2) creates devices
	(3) prints device information
	(4) checks connection for device
	(5) prints connection information
	'''
	device_info = system.device_infos
	devices = system.create_device()
	# for each device: print all attributes
	for i, info in enumerate(device_info):
		print(f"{TAB1}Device {i}:")
		for item in info:
			print(f"{TAB2}{item}: {info[item]}")
		# check connection and print connection info if device is connected
		if devices[i].is_connected():
			print(f"{TAB2}connection: True")
			pixel_format = str(devices[i].nodemap.get_node("PixelFormat").value)
			frame_rate = str(round(devices[i].nodemap.get_node("AcquisitionFrameRate").value, 1))
			print(f"{TAB2}pixel format: {pixel_format}")
			print(f"{TAB2}frame rate: {frame_rate}fps")
		else:
			print(f"{TAB2}connection: False")

def addUnicastDevice():
	'''
	adds unicast device(s)
	(1) enumerates devices before adding unicast device
	(2) takes IP address for device to be added from user
	(3) adds unicast discovery device(s)
	(4) enumerates devices after adding unicast device(s)
	'''
	
	# enumerate devices before adding unicast device(s) 
	print(f"{TAB1}Device list before adding unicast device(s)")
	enumerate_devices()

	# stay in loop until exit
	while True:
		ip = input(f"{TAB1}Input IP for device to be added ('x' to exit): ")
        # exit manually on 'x'
		if ip.__eq__('x'):
			print(f"{TAB1}Succesfully exited")
			break
		
		'''
		Add a unicast discovery device
			registers an IP address for a device on a different subnet 
			than the host. Registered devices will be enumerated using 
			unicast discovery messages. The list of remote devices will 
			persist until they are removed using RemoveUnicastDiscoveryDevice() 
			or until the application terminates. Unicast discovery's will be 
			sent when UpdateDevices() is called.
		'''
		print(f"{TAB1}Add device with ip: {ip}")
		system.add_unicast_discovery_device(ip)
	
	# enumerate devices after adding unicast device(s) 
	print(f"{TAB1}Device list after adding unicast device(s)")
	enumerate_devices()

def example_entry_point():
	# run example
	addUnicastDevice()
	
	# clean up example
	'''
	Explicit destroy: unnecessary.
		Automatically called when module closes. Also: cannot reopen, as there is
		no explicit open.
	'''
	system.destroy_device()
	print(f'{TAB1}Destroyed all created devices')

if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
