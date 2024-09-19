#!/bin/bash
set -e

# default venv name is venv
venv_path="${1:-venv}"

python3 -m venv $venv_path

source $venv_path/bin/activate

pip install -U pip setuptools wheel

pip install -e .

# OPTIONAL: For CUDA capable machines
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

