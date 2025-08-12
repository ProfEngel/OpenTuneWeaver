#!/bin/bash

# Conda Environment Setup für OpenTuneWeaver

# Installiere Miniconda falls nicht vorhanden
if ! command -v conda &> /dev/null; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    source ~/.bashrc
fi

# Erstelle Conda Environment
conda create -n otw python=3.10 -y
conda activate otw

# Installiere CUDA Toolkit
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9 -y

# Installiere PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# System-Tools
sudo apt-get update
sudo apt-get install -y cmake build-essential libreoffice wkhtmltopdf

sudo apt update

sudo apt upgrade -y

sudo apt install -y build-essential git wget curl

sudo apt install libcurl4-openssl-dev

sudo apt install -y cmake

# Python Dependencies
pip install -r requirements.txt

# Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Baue llama.cpp
cd pipeline/modules/06_finetuning
git clone --recursive https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j
cd ../../../../

# Starte App
cd ui
python app.py