# Biodiversity Foundation Model (BFM)

First steps towards developing an Artificial Intelligence Foundation Model for Biodiversity research and conservation.

This repository contains a self-contained implementation of the underlying architecture of the BFM.

## Installation

There are 2 ways to install the software:

This software is tested to work with Python 3.10 and 3.12

1) With pip

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
# from setuptools 61 onwards, it's possible to install with pip from a pyproject.toml
pip install -e .
# OPTIONAL: For CUDA capable machines
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**OR**

2) With poetry. (Make sure you have [Poetry](https://python-poetry.org/docs/#installation) installed)

To install poetry, you can simply run
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Afterwards, just run in a terminal
```bash
poetry install
```
To run the scripts, activate the virtual env
```bash
poetry shell
```

## Run experiments

Start an MLFLow server
```bash
# change here port if you get [Errno 98] Address already in use
# also change port in the bfm_model/bfm/configs
# & - optional, for interactive mode
mlflow server --host 0.0.0.0 --port 8082 [&]
```
On another terminal (if not running in interactive mode), run the train recipe
```bash
python bfm_model/bfm/train.py
```

In case of running the AQFM training script, make sure that the port you defined in it corresponds to the port of the MLFlow running server.

## Connection

```bash
# start an interactive job
salloc -p gpu_h100 --gpus-per-node=1 -t 01:00:00
# ssh to the node with port forwarding
# ssh gcn140


# Forwarding the node mlflow instance to the local machine
# N.B.: Make sure to specify the host on the local machine, as specifying just the port might results in "Permission denied" errors.
# N.B.2.: If specifying the host 0.0.0.0 on the local machine, access by using `localhost:<port_id>`.
ssh -i .ssh/snelius_key -L 0.0.0.0:<desired_port_on_local>:[gcn|tcn]<node_id>:<mlflow_port_on_remote> <user_name>@snellius.surf.nl
ssh -L 0.0.0.0:8083:gcn112:8082 mmensio1@snellius.surf.nl

```

## Running via scheduled job

```bash
sbatch job.sh
```

Then you can observe mlflow with the same bind command:
```bash
ssh -i .ssh/snelius_key -L 0.0.0.0:<desired_port_on_local>:gcn<node_id>:<mlflow_port_on_remote> <user_name>@snellius.surf.nl
```


## OOM Errors:

If experience any, use: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

An interesting discussion: https://github.com/pytorch/pytorch/issues/122057

Issue PyTorch 2.1.2 vs 2.2.0

Good Trainer example: https://github.com/SeanNaren/min-LLM/blob/fsdp/train.py#L235-L243

Interesting addition for CLI args generation: https://github.com/google/python-fire


## Resources

+ [FSDP Lighting](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)


## TODODs

[ ] Include the species_extinction_variables into the Encoder's latents and embeddings. Currently swaped with the species distribution - which have an extra dim, that needs handling.

[ ] Fix the configurable patch size:
```
#TODO Check why this gives weird error. For now hardcode the # of patches
        # num_patches = (H // self.patch_size) * (W // self.patch_size)
        num_patches = 3040
```
