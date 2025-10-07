#!/usr/bin/env bash
# Create a conda env compatible with the document's versions
ENV_NAME=asc-iot-env
conda create -y -n $ENV_NAME python=3.7
conda activate $ENV_NAME
conda install -y -c conda-forge librosa==0.9.2
pip install -r requirements.txt
echo "Activate with: conda activate $ENV_NAME"
