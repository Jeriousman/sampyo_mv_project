#!bin/bash
 
git clone https://github.com/facebookresearch/detectron2.git 
cd detectron2
pip install -e .

cd ..
git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh

pip install fvcore
