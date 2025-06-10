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
Enumeration: CcpSwitchover
    This example introduces device enumeration with the ability to hand over control
    to another process. This includes opening and closing the system, updating and 
    retrieving the list of devices, searching for devices, and creating and destroying
    a device. In this example, we will also set a special key to the device that another 
    process can use to aquire control of the device when running the example for the first
    time. Running the example a second time while the first instance is still running will 
    try to use the special key to gain control of the device. 	
'''  
TAB1 = "  "
TAB2 = "    "

'''
=-=-=-=-=-=-=-=-=-
=-=- SETTINGS =-=-
=-=-=-=-=-=-=-=-=-
'''

def create_device_info_with_tries():
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		device_infos = system.device_infos
		if not device_infos:
			print(
				f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'{TAB1}secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				print(f'{TAB1}{sec_count + 1 } seconds passed ',
					'.' * sec_count, end='\r')
			tries += 1
		else:
			print(f"{TAB1}Enumerate Device")
			return device_infos
	else:
		raise BaseException(f'No device found! Please connect a device and run '
						f'the example again.')


def enumerate_device():
	'''
	demonstrates enumeration
	(1) gets device info list
	(2) save first serial number to demonstrate search
	(3) prints device information
	(4) demonstrates device search
	(5) create device
	(6a) if running first instance will set a special key
	(6b) if running second instance will use key to aquire control
	(7) cleans up
	'''

	device_infos = create_device_info_with_tries()
	serial_to_find = "00000"
	for i, device_info in enumerate(device_infos):
		if i == 0:
			serial_to_find = device_info['serial']
		
		print(f"{TAB2}Information for device {i}:")
		vendor = device_info['vendor']
		model = device_info['model']
		serial = device_info['serial']
		macStr = device_info['mac']
		ipStr = device_info['ip']

		print(f"{TAB2}Vendor: {vendor}; model: {model}; serial: {serial}; MAC: {macStr}; ip: {ipStr}")
	
	print(f"{TAB1}Searching for device with serial number: {serial_to_find}")

	matching_device_info = None
	for info in device_infos:
		if info['serial'] == serial_to_find:
			matching_device_info = info
			break
	
	if matching_device_info:
		print(f"{TAB1}Device found!")

		'''
		Create device
			Create device in order to configure it and grab images. A device can only be created
			once per process, and can only be opened with read-write access once. 
		'''

		print(f"{TAB1}Creating device")

		devices = system.create_device(matching_device_info)
		device = devices[0]

		# static key used to acquire control between applications
		switchoverKey = 0x1234

		if (has_control(device)):
			device.tl_device_nodemap['CcpSwitchoverKey'].value = switchoverKey
			print(f"{TAB1}Please input a character to continue: ", end = "")
			input()
		else:
			device.tl_device_nodemap['CcpSwitchoverKey'].value = switchoverKey
			device.tl_device_nodemap['DeviceAccessStatus'].value = "ReadWrite"
			if (has_control(device)):
				print(f"{TAB1}Create device succeeded with acquiring control")
			else:
				print(f"{TAB1}Create device failed to acquire control")

		system.destroy_device()
		print(f"{TAB1}Destroyed all created devices")

def has_control(device):
	# Check if device has control
	device_access_status = device.tl_device_nodemap['DeviceAccessStatus'].value
	return device_access_status == "ReadWrite"

if __name__ == '__main__':
	print('\nExample started\n')
	enumerate_device()	
	print('\nExample finished successfully')

