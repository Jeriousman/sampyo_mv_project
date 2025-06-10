from requests.auth import HTTPDigestAuth
import requests
from onvif import ONVIFCamera
import datetime

def get_snapshot_and_save():

    my_cam = ONVIFCamera('192.168.200.3', 80, 'admin', 'sampyo123!')
   
    media_service = my_cam.create_media_service()
    
    media_profile = media_service.GetProfiles()[0]
    
    snapshot_uri = media_service.GetSnapshotUri({'ProfileToken': media_profile.token}).Uri
    print(datetime.datetime.now())
    response = requests.get(snapshot_uri, auth=HTTPDigestAuth('admin','sampyo123!'))
    if response.status_code == 200:
        file_path = f'{datetime.datetime.now()}.jpg'
        #file_path = "a.jpg"
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print("Snapshot saved successfully at", file_path)
    else:
        print("Failed to fetch snapshot")
if __name__ == "__main__":
    get_snapshot_and_save()

