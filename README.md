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

### Training

```bash
python bfm_model/bfm/train_lighting.py
```

### Testing

```bash
python bfm_model/bfm/test_lighting.py
```

### Rollout Predictions

```bash
python bfm_model/bfm/rollouts.py
```

### Rollout Finetuning

```bash
python bfm_model/bfm/rollout_finetuning.py
```

## Analysing results

We use [Hydra](https://hydra.cc/docs/intro/) to store all the artifacts from all the runs. In this way we can configure with yaml files, override from CLI, make multiruns with multiple parameters, and have all the results stored in the `outputs` folder.
There, we can find by date and time all the data from the runs (configs, checkpoints, metrics, ...).


### How to use MLflow

[MLflow](https://mlflow.org/docs/latest/index.html) is used to log all the runs, and we configure it to save its internal files in the `mlruns` folder. The logging is done via filesystem, so that you don't need to have a MLflow server running during the training.

You can run the MLflow server to inspect the runs with the command:

```bash
# you can customize host and port depending on your system
mlflow server --host 0.0.0.0 --port 8082
```

On snellius, you need to forward the ports to your machine (TODO document commands)



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

## Resources

+ [FSDP Lighting](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)

+ Good Trainer example: https://github.com/SeanNaren/min-LLM/blob/fsdp/train.py#L235-L243

+ Interesting addition for CLI args generation: https://github.com/google/python-fire

## TODODs
[ ] Finetune routine implementation with LoRA and optinally VeRA

[X] Finetune dataset setup

[ ] Rollout Finetune modes: Daily (4x6h) - Weekly & Monthly

[ ] Investigate if a (Prioritized) Buffer for Rollout Finetune is required

a) Presence and absence of species: [Geolifeclef](https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/data
)
b) Invasive species [flavonge](https://floraveg.eu/) & [opendap](http://opendap.biodt.eu/ias-pdt/0/outputs/)

[X] Validate distributed training strategy

[ ] 
