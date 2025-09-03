sudo apt install python3.10
sudo apt install python3.10-venv
python3.10 -m venv .venv
source .venvs/bin/activate



tar -xvzf ./toinstall/ArenaSDK_v0.1.91_Linux_x64.tar.gz -C .
cd ArenaSDK_Linux_x64
sudo sh Arena_SDK_Linux_x64.conf
cd ..

unzip toinstall/arena_api-2.5.9-py3-none-any.zip
pip install -r examples/requirements_lin.txt
pip install arena_api-2.5.9-py3-none-any.whl
sudo apt-get install python3-tk

pip install torch==2.5.1 
pip install torchvision==0.20.1 
pip installtorchaudio==2.5.1

git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
cd ..
git clone git@github.com:facebookresearch/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..


git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../..

pip install depthai
pip install blobconverter


