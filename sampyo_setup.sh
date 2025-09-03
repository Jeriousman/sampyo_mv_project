

source hojun_venv/bin/activate


pip install torch==2.5.1 
pip install torchvision==0.20.1 
pip install torchaudio==2.5.1


### Installing Lucid Arena
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



### Installing Luxonis DepthAI
sudo wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash
echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
source ~/.bashrc
python3 -m pip install depthai


git clone git@bitbucket.org:sdt_inc/sampyo-hwaseong.git
mv sampyo-hwaseong onvif
cd onvif
git checkout dev2

mv /home/sdt/Workspace/onvif/python-onvif-zeep /home/sdt/Workspace/onvif/python-onvif-zeep_
git clone https://github.com/FalkTannhaeuser/python-onvif-zeep
pip install zeep
cd python-onvif-zeep 
python3 setup.py install
mv /home/sdt/Workspace/onvif/python-onvif-zeep_/* /home/sdt/Workspace/onvif/python-onvif-zeep/
rm -rf /home/sdt/Workspace/onvif/python-onvif-zeep_




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
cp -f /home/sdt/Workspace/onvif/change_lib_files/defaults.py /home/sdt/Workspace/onvif/python-onvif-zeep/socket/detectron2/detectron2/engine/defaults.py

pip install fvcore
cd /home/sdt/Workspace/onvif/python-onvif-zeep/socket
git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
cd /home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/demo/
gdown https://drive.google.com/uc?id=17lbV4jHBSrKc5Qgg179_3898uGCsThlj

cp -f /home/sdt/Workspace/onvif/change_lib_files/Base-COCO-InstanceSegmentation.yaml /home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/configs/coco/instance-segmentation/
cp -f /home/sdt/Workspace/onvif/change_lib_files/maskdino_R50_bs16_50ep_3s.yaml /home/sdt/Workspace/onvif/python-onvif-zeep/socket/MaskDINO/configs/coco/instance-segmentation/


cd /home/sdt/Workspace/onvif/python-onvif-zeep/socket
mv /home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2 /home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2_
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints
./download_ckpts.sh
mv /home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2_/* /home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/
rm -rf /home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2_
cd ../..

pip install blobconverter
pip install pyodbc
pip install requests
pip install matplotlib
pip install pillow
pip install timm
pip install pandas



mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/result
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/between_25_40_img
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/combined_mask_np
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_array
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_depth_img
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_mask
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_max
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/depth_rgb
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/interested_area
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/mv_rgb
mkdir /home/sdt/Workspace/onvif/python-onvif-zeep/hojun/image_test/sam_img
mkdir /home/sdt/Workspace/onvif/image_bucket