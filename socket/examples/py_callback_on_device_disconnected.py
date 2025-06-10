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

from arena_api.callback import callback, callback_function
from arena_api.system import system

'''
Callbacks: On Device Disconnected
	This example demonstrates how to register a callback to get notified when a
	device has disconnected. At first this example will enumerate devices then if
	there is any device found it will register a disconnect callback for a
	discovered device. Next the program will wait until a user inputs an exit
	command. While this example waits for input, feel free to disconnect the
	device. When the device is disconnected the callback will be triggered and it
	will print out info of the device that was removed by using
	print_disconnected_device_info function.
'''
TAB1 = "  "
TAB2 = "    "

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


# Must have the decorator on the callback function
@callback_function.system.on_device_disconnected
def print_disconnected_device_info(device):
	'''
	Print information from the callback
		When registered device is disconnected System fire a callback which pass
		disconnected device to this function.
	'''
	print(f'\n{TAB1}Device was disconnected:\n{TAB2}{device}')
	print(f'{TAB1}Press <ENTER> to continue')

def example_entry_point():
	# Create a device
	devices = create_devices_with_tries()
	device = system.select_device(devices)

	# -------------------------------------------------------------------------
	# Register a disconnect callback test

	handle = callback.register(
		system, print_disconnected_device_info, watched_device=device)

	input(f'{TAB1}Waiting for user to disconnect a device'
		f' or press <ENTER> to continue')

	print(f'{TAB1}Check if device is connected:')
	if device.is_connected() is False:
		print(f'{TAB2}Device is disconnected')
	else:
		print(f'{TAB2}Device is connected')

	# -------------------------------------------------------------------------
	# Clean up

	# Deregister an individual disconnect callback
	callback.deregister(handle)

	# destroy device (optional, implicit)
	system.destroy_device()
	print(f'{TAB1}Destroyed all created devices')


if __name__ == '__main__':
	print('Example started\n')
	example_entry_point()
	print('\nExample finished successfully')
