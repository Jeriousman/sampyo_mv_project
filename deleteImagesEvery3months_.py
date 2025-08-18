import os
import time
from datetime import datetime
import shutil
## 삭제할 폴더 경로 


for dirpath, dirnames, filenames in os.walk("/home/sdt/Workspace/onvif/hojun_test"):
    if dirpath=="/home/sdt/Workspace/onvif/hojun_test":
        continue
    else:
        print(dirpath)
        shutil.rmtree(dirpath)
        os.makedirs(dirpath, exist_ok=True)
        os.makedirs(dirpath+'aa', exist_ok=True)
        os.makedirs(dirpath+'bb', exist_ok=True)

        
