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

from pprint import pprint, PrettyPrinter
import json

from arena_api.system import system

'''
Force IP
	This example demonstrates how to force network settings. It does this by
	adding 1 to the final octet of the IP address. It leaves the subnet mask and
	default gateway unchanged, although the same method is used to change these as
	well.
'''
TAB1 = "  "
TAB2 = "    "

def add_one_to_ip(ip):

	octet0, octet1, octet2, octet3 = ip.split('.')
	if octet3 == '254':  # Avoid 255
		octet3 = '1'
	else:
		octet3 = str(int(octet3) + 1)
	return f'{octet0}.{octet1}.{octet2}.{octet3}'

pp = PrettyPrinter()

def print_formatted_device_info(elem):
	if elem is not None:
		dict_str = json.dumps(elem, indent=2)
		formatted_str = ''.join([TAB2 + TAB1 + line for line in dict_str.splitlines(True)])
		print(formatted_str)

def select_device_info(device_infos):

	if isinstance(device_infos, list):
		if len(device_infos) == 0:
			raise ValueError('input can not be an empty list')
		elif not all(isinstance(d, dict) for d in device_infos):
			raise TypeError(f'Expected list of dictionary '
                            f'instead of {type(device_infos).__name__}')
	else:
		raise TypeError(f'Expected list of dictionary '
                        f'instead of {type(device_infos).__name__}')
			
	num_device_infos = len(device_infos)
	selection = 0
	
	if num_device_infos == 1:
		print_formatted_device_info(device_infos[0])
		print(f'{TAB2}Automatically selecting this device info.')
		return device_infos[0]
        
	print(f'{TAB1}Select device info:')
	for num, device_info in enumerate(device_infos):
		print(f'{TAB2}{num+1}.')
		print_formatted_device_info(device_info)
		
	while True:
		try:
			selection = int(input(TAB2 + 'Make selection (1-' + str(num_device_infos) + '): '))
		except ValueError:
			print(f'{TAB2}Invalid device info selected, please select a device in range.')
			continue

		if 0 < selection and selection <= num_device_infos:
			selected_device_info = device_infos[selection-1]
			break
		else:
			print(f'{TAB2}Invalid device info selected, please select a device in range.')

	return selected_device_info

def example_entry_point():

	# Discover devices --------------------------------------------------------

	print(f'{TAB1}Discover devices on network')
	device_infos = system.device_infos
	print(f'{TAB1}{len(device_infos)} devices found')

	if not device_infos:
		raise BaseException('No device is found!')

	# Choose the first device for this example
	device_info = select_device_info(device_infos)

	'''
	Forcing the IP address requires a device's MAC address to specify the
		device. This example grabs the IP address, subnet mask, and default
		gateway as well to display changes and return the device to its original
		IP address.
	'''
	print(f'{TAB1}Selected device info: ')
	print_formatted_device_info(device_info)

	# Create new IP -----------------------------------------------------------

	print(f'{TAB1}Current IP = ', device_info['ip'])
	new_ip = add_one_to_ip(device_info['ip'])

	# The original device_info can be used however system.force_ip
	# will used on 'mac' ,'ip' ,'subnetmask' , and 'defaultgateway'.
	# This new dictionary to show what is needed by 'System.force_ip()'
	device_info_new = {
		'mac': device_info['mac'],
		'ip': new_ip,
		'subnetmask': device_info['subnetmask'],
		'defaultgateway': device_info['defaultgateway']
	}
	print(f'{TAB1}New IP = ', device_info_new['ip'])

	# Force IP ----------------------------------------------------------------

	'''
	Note: The force_ip function can also take a list of device infos to
		force new IP addesses for multiple devices.
	'''
	print(f'{TAB1}New IP is being forced')
	system.force_ip(device_info_new)
	print(f'{TAB1}New IP was forced successfully')


if __name__ == '__main__':
	print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
	print('\nExample started\n')
	example_entry_point()
	print('\nExample finished successfully')
