# test PC: 192.168.1.104 / 10.110.31.139
from time import sleep
from onvif import ONVIFCamera
import zeep
from requests.auth import HTTPDigestAuth
import requests
import datetime
import time

IP="192.168.200.3"   # Camera IP address
PORT=80           # Port
USER="admin"         # Username
PASS="sampyo123!"        # Password

XMAX = 1
XMIN = -1
YMAX = 1
YMIN = -1

# snapshot
def get_snapshot_and_save(media):
    start = datetime.datetime.now()
    cam_open = datetime.datetime.now()
    print('cam open time : ',cam_open - start)
    service_open = datetime.datetime.now()
    print('service open time : ', service_open-cam_open)
    media_profile = media.GetProfiles()[0]

    snapshot_uri = media.GetSnapshotUri({'ProfileToken': media_profile.token}).Uri
    file_path = f'snapshot_{datetime.datetime.now()}.jpg'
    start_rsp = datetime.datetime.now()
    response = requests.get(snapshot_uri, auth=HTTPDigestAuth('admin','sampyo123!'))
    print(response)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print("Snapshot saved successfully at", file_path)
        print('requrest time : ', datetime.datetime.now() - start_rsp)
    else:
        print("Failed to fetch snapshot")


def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue


def perform_move(ptz, request, timeout):
    # Start continuous move
    ptz.ContinuousMove(request)
    # Wait a certain time
    sleep(timeout)
    # Stop continuous move
    ptz.Stop({'ProfileToken': request.ProfileToken})


def move_up(ptz, request, timeout=1):
    print('move up...')
    request.Velocity.PanTilt.x = 0
    request.Velocity.PanTilt.y = YMAX
    perform_move(ptz, request, timeout)


def move_down(ptz, request, timeout=1):
    print('move down...')
    request.Velocity.PanTilt.x = 0
    request.Velocity.PanTilt.y = YMIN
    perform_move(ptz, request, timeout)


def move_right(ptz, request, timeout=1):
    print('move right...')
    request.Velocity.PanTilt.x = XMAX
    request.Velocity.PanTilt.y = 0

    perform_move(ptz, request, timeout)


def move_left(ptz, request, timeout=1):
    print('move left...')
    request.Velocity.PanTilt.x = XMIN
    request.Velocity.PanTilt.y = 0
    perform_move(ptz, request, timeout)


def zoom_up(ptz, request, timeout=1):
    print('zoom up')
    request.Velocity.Zoom.x = 1
    request.Velocity.PanTilt.x = 0
    request.Velocity.PanTilt.y = 0
    perform_move(ptz, request, timeout)


def zoom_down(ptz, request, timeout=1):
    print('zoom down')
    request.Velocity.Zoom.x = -1
    request.Velocity.PanTilt.x = 0
    request.Velocity.PanTilt.y = 0
    perform_move(ptz, request, timeout)


def move_abspantilt(ptz, pan, tilt, requesta, velocity=1):
    requesta.Position.PanTilt.x = pan
    requesta.Position.PanTilt.y = tilt
    requesta.Speed.PanTilt.x = velocity
    requesta.Speed.PanTilt.y = velocity
    ret = ptz.AbsoluteMove(requesta)

def get_current_pan_tilt(ptz, media_profile):
    status = ptz.GetStatus({'ProfileToken': media_profile.token})
    print('pan: ', round(status.Position.PanTilt.x,3), 'tilt: ', round(status.Position.PanTilt.y,3))
    #print('pan: ',status.Position.PanTilt.x,'tilt: ',status.Position.PanTilt.y)

def continuous_move():
    mycam = ONVIFCamera(IP, PORT, USER, PASS)
    # Create media service object
    media = mycam.create_media_service()
    # Create ptz service object
    ptz = mycam.create_ptz_service()

    # Get target profile
    zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
    media_profile = media.GetProfiles()[0]

    # Get PTZ configuration options for getting continuous move range
    request = ptz.create_type('GetConfigurationOptions')
    request.ConfigurationToken = media_profile.PTZConfiguration.token
    ptz_configuration_options = ptz.GetConfigurationOptions(request)

    # AbsoluteMove
    requesta = ptz.create_type('AbsoluteMove')
    requesta.ProfileToken = media_profile.token
    requesta.Position = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
    requesta.Speed = ptz.GetStatus({'ProfileToken': media_profile.token}).Position


	
    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = media_profile.token
    ptz.Stop({'ProfileToken': media_profile.token})

    if request.Velocity is None:
        request.Velocity = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
        request.Velocity = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
        request.Velocity.PanTilt.space = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].URI
        request.Velocity.Zoom.space = ptz_configuration_options.Spaces.ContinuousZoomVelocitySpace[0].URI

    # Get range of pan and tilt
    # NOTE: X and Y are velocity vector
    global XMAX, XMIN, YMAX, YMIN
    XMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
    XMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min
    YMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
    YMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min

    while True:
        action = input("Enter the action ('u', 'd', 'l', 'r', 'zi', 'zo','m','status', 's', 't' or 'q'): ").strip().lower()

        if action == "u":
            move_up(ptz, request)
        elif action == "d":
            move_down(ptz, request)
        elif action == "l":
            move_left(ptz, request)
        elif action == "r":
            move_right(ptz, request)
        elif action == "zi":
            zoom_up(ptz, request)
        elif action == "zo":
            zoom_down(ptz, request)
        elif action =='m':
            pan = input("pan : ").strip().lower()
            tilt = input("tilt : ").strip().lower() 
            move_abspantilt(ptz,pan, tilt, requesta)
        elif action == "s":
            get_snapshot_and_save(media)
        elif action == "status":
            get_current_pan_tilt(ptz, media_profile)
        elif action == "t":
            get_snapshot_and_save(media)
            snap_1 = time.time()
            move_abspantilt(ptz, 1, -0.05, requesta)
            move = time.time()
            get_snapshot_and_save(media)
            snap_2 = time.time()
            print("move and capture : ", snap_2 - snap_1)
        elif action == "q":
            break
        else:
            print("Invalid action. Please enter 'u', 'd', 'l', 'r', 'zi', 'zo','m', 's', 't' or 'q'.")


if __name__ == '__main__':
    continuous_move()

