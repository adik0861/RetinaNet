# Copy dataset over
cd
gsutil -m cp -r gs://visdrone .
mv visdrone data


# Install CUDA drivers
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda


# Setup Anaconda
mkdir downloads
cd downloads
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
echo "export PATH=~/anaconda3/bin:$PATH" >> ~/.bashrc
source ~/.bashrc

# Either git clone or upload RetinaNet as it contains the env files for conda
cd
git clone https://github.com/adik0861/RetinaNet.git
cd ~/RetinaNet
conda env create -f environment.yml
conda activate torch

# Install pytorch
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install nvidia apex using the patched setup.py
git clone https://github.com/NVIDIA/apex
cd apex
cp ~/RetinaNet/utilities/nvidia_apex_patch/setup.py ~/downloads/apex/
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
