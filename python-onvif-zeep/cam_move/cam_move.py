from time import sleep
from onvif import ONVIFCamera
import zeep
from requests.auth import HTTPDigestAuth
import requests
import datetime
import time
import traceback
import logging.handlers
import json

######################################################################
#                             Save Log                               #
######################################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
        filename=f"./logs/cam_move.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.addHandler(log_fileHandler)


######################################################################
#                              Config                                #
######################################################################
with open('./config/ptz_info.json', 'r') as f:
    cfg = json.load(f)

IP = cfg['ip']
PORT = cfg['port']
USER = cfg['user']
PASS = cfg['password']
PAN = '0.98'
TILT = '-0.03'


def move_abspantilt(ptz, pan, tilt, requesta, velocity=1):
    requesta.Position.PanTilt.x = pan
    requesta.Position.PanTilt.y = tilt
    requesta.Speed.PanTilt.x = velocity
    requesta.Speed.PanTilt.y = velocity
    ret = ptz.AbsoluteMove(requesta)

def get_current_pan_tilt(ptz, media_profile):
    status = ptz.GetStatus({'ProfileToken': media_profile.token})
    return ('pan: ', round(status.Position.PanTilt.x,3), 'tilt: ', round(status.Position.PanTilt.y,3))


def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue

def set_camera():
    mycam = ONVIFCamera(IP, PORT, USER, PASS)
    # Create media service object
    media = mycam.create_media_service()
    # Create ptz service object
    ptz = mycam.create_ptz_service()

    # Get target profile
    zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
    media_profile = media.GetProfiles()[0]

    # AbsoluteMove
    requesta = ptz.create_type('AbsoluteMove')
    requesta.ProfileToken = media_profile.token
    requesta.Position = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
    requesta.Speed = ptz.GetStatus({'ProfileToken': media_profile.token}).Position

    return ptz, requesta, media_profile



if __name__ == '__main__':
    while True:
        try:
            ptz, requesta, media_profile = set_camera()
            move_abspantilt(ptz, PAN, TILT,requesta)
            status = get_current_pan_tilt(ptz, media_profile)
            logger.info(f"change to {status}")
            break
        except Exception as e:
            logger.error(traceback.format_exc())



