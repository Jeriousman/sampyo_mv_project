import os
import time
from datetime import datetime
import shutil
## 삭제할 폴더 경로 
ptz_images_path="/home/sdt/Workspace/onvif/image_bucket"
mv_depth_images=path="/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test"




## PTZ 이미지 저장된 디렉토리 모두 삭제 
for dirpath, dirnames, filenames in os.walk(ptz_images_path):
    if dirpath == "/home/sdt/Workspace/onvif/image_bucket":
        continue
    else:
        print(dirpath)
        shutil.rmtree(dirpath)

for dirpath, dirnames, filenames in os.walk(mv_depth_images):
    if dirpath=="/home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test":
        continue
    else:
        print(dirpath)
        shutil.rmtree(dirpath)
        os.makedirs(dirpath, exist_ok=True)
        
