#!/bin/bash
# Most of the commands I had to run to get PyTorch and PyVision working. 
wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base
pip3 install Cython
pip3 install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl

pip3 install setuptools
pip3 install --upgrade setuptools
pip3 install wheel
pip3 install torch torchvision

git clone https://github.com/pytorch/vision
cd vision

sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libjpeg-dev zlib1g-dev

python setup.py install
