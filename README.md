# Biodiversity Foundation Model (BFM)

First steps towards developing an Artificial Intelligence Foundation Model for Biodiversity research and conservation.

This repository contains a self-contained implementation of the underlying architecture of the BFM. 

## Installation

There are 2 ways to install the software:

This software is tested to work with Python 3.10 and 3.12

1) With pip

```
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
# OPTIONAL: For CUDA capable machines
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**OR**

2) With poetry. (Make sure you have [Poetry](https://python-poetry.org/docs/#installation) installed)

Just run in a terminal
```
poetry install
```
To run the scripts, activate the virtual env
```
poetry shell
```
To add and export dependecies use:
```
poetry add <dependency>
poetry export -f requirements.txt --without-hases > requirements.txt
```

## Run experiments

Start an MLFLow server
```bash
# change here port if you get [Errno 98] Address already in use
# also change port in the src/bfm/src/configs
mlflow server --host 127.0.0.1 --port 8081
```
On another terminal, run the train recipe
```
python src/bfm/src/train.py
```

## TODOs

- [ ] Export new requirements.txt
- [ ] Make the output folder system coherent
- [ ] Add more logging points
- [ ] Add checkpointing 
- [ ] Test multi-node, multi-gpu runs
- [ ] Cleanup, remove prints etc.