# This is for running on cloudlab

sudo apt-get install python3-venv
sudo apt-get install python3-pip

python3 -m venv env
source env/bin/activate

pip install -r greatlakes/basic_pretrained_sae_profiling_one_gpu/requirements.txt
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# nvidia-smi
sudo apt update
sudo apt install nvidia-driver-470 nvidia-utils-470
sudo reboot
