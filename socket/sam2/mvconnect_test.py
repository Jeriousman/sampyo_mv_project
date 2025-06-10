import time
from arena_api.system import system
import traceback

try:
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
            break
    else:
        raise Exception(f'No device found! Please connect a device and run '
                        f'the example again.')

    device = system.select_device(devices)
    nodemap = device.nodemap
    tl_stream_nodemap = device.tl_stream_nodemap
    print(f'Device used in the example:\n\t{device}')

    #print('device.DEFAULT_NUM_BUFFERS: ',device.DEFAULT_NUM_BUFFERS)
    device_status = device.tl_device_nodemap
    print(device_status['DeviceAccessStatus'].value)

except:
    print(traceback.format_exc())
finally:
    device.stop_stream()
    system.destroy_device(device)
    print('device 해제 완료')