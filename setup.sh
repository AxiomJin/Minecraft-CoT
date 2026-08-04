git clone --recurse-submodules https://github.com/CraftJarvis/OpenHA.git
conda create -n openha python=3.10
conda activate openha
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda install --channel=conda-forge openjdk=8 -y
pip install -e .
