#!/bin/bash
export DEPTHAI_WATCHDOG_INITIAL_DELAY=60000
export DEPTHAI_BOOTUP_TIMEOUT=60000
source /home/sdt/Workspace/onvif/python-onvif-zeep/socket/.venvs/sampyo/bin/activate
python3 -V
python3 /home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/server_socket_test_depth.py
