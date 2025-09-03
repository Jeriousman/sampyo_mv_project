

source hojun_venv/bin/activate



tar -xvzf ArenaSDK_v0.1.91_Linux_x64.tar.gz -C .
cd ArenaSDK_Linux_x64
sudo sh Arena_SDK_Linux_x64.conf
cd ..

unzip arena_api-2.5.9-py3-none-any.zip
pip install -r examples/requirements_lin.txt
pip install arena_api-2.5.9-py3-none-any.whl
sudo apt-get install python3-tk
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84


pip install torch==2.5.1 
pip install torchvision==0.20.1 
pip install torchaudio==2.5.1


git clone https://github.com/FalkTannhaeuser/python-onvif-zeep
pip install zeep
cd python-onvif-zeep 
python3 setup.py install
mkdir -p /home/sdt/Workspace/onvif/python-onvif-zeep/socket/weights
cd /home/sdt/Workspace/onvif/python-onvif-zeep/socket/weights
pip install gdown
gdown https://drive.google.com/uc?id=1Vm9yL5NASDYzEYqwkESJXMhvtmH2NtC7
gdown https://drive.google.com/uc?id=1FUmterEMEe6kv-j4dQjd5BeLuCq26xzN

cd /home/sdt/Workspace/onvif/python-onvif-zeep/socket
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

pip install fvcore
cd /home/sdt/Workspace/onvif/python-onvif-zeep/socket
git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
cd /home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/demo/
gdown https://drive.google.com/uc?id=17lbV4jHBSrKc5Qgg179_3898uGCsThlj


cd ../../../../..
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../..

pip install depthai
pip install blobconverter
pip install pyodbc
pip install requests
pip install matplotlib
pip install pillow
pip install timm
pip install pandas



