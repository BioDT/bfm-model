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
salloc -p gpu_h100 --nodes 1 --gpus-per-node 2 -t 02:00:00
source venv/bin/activate
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

We offer 2 Parameter Efficient Finetuning Techniques, namely LoRA and VeRA. They can be configured by enabling and disabling interchangable each of them on the `train_config.yaml` on the `finetune` section.

```bash
python bfm_model/bfm/rollout_finetuning.py
```

### In the cluster
```bash
sbatch snellius_train.sh
# or
sbatch snellius_finetune.sh
```

## Analysing results

We use [Hydra](https://hydra.cc/docs/intro/) to store all the artifacts from all the runs. In this way we can configure with yaml files, override from CLI, make multiruns with multiple parameters, and have all the results stored in the `outputs` folder.
There, we can find by date and time all the data from the runs (configs, checkpoints, metrics, ...).


### MLflow

[MLflow](https://mlflow.org/docs/latest/index.html) is used to log all the runs, and we configure it to save its internal files in the `mlruns` folder. The logging is done via filesystem, so that you don't need to have a MLflow server running during the training.

You can run the MLflow server when you want (after or during training) to inspect the runs with the command:

```bash
# run in the root of the repository, where the mlruns folder is located
mlflow server --host 0.0.0.0 --port 8082
```

On snellius:
- run the `mlflow` command above in the same node where your vscode interface is executing (login node or ondemand)
- vscode will detect the port and forward a local port to it (popup appearing, or go to the "PORTS" tab to open it)

If you are not using vscode, or want a manual connection:
- forward a local port to it: `ssh -L 0.0.0.0:<LOCAL_PORT>:<node_id>:8082 <USER>@snellius.surf.nl` (example: `ssh -L 0.0.0.0:8899:int6:8082 snellius`)
- open `http://localhost:<LOCAL_PORT>/` (example: `http://localhost:8899/`)

## Visualisation
This repository contains various visualisation functions that are applicable for every stage of the workflow. More specific:

- **Batch level:** Inspect and visualise the RAW data (2 timesteps) from the Batches along with their MAE. Run the notebook `documentation/batch_visualisation.ipynb`. You need to change the `DATA_PATH` to the directory you have the batches you want to visualise. The code plots only a single batch but it can be configured to visualise all of them and save them with the appropriate flag.

> [!NOTE]
> You need to produce predictions either by running `bfm_model/bfm/test_lighting.py` or by `bfm_model/bfm/rollout_finetuning.py` and enabling the **finetune.prediction: True** on the train_config. These will create export folders with the predictions and the ground truths in a compact tensor format.

- **Prediction level:** To visualise them simply run `streamlit run prediction_viewer.py`. You can navigate the different tabs and variable groups to inspect each and every one of them.

- **Rollout level:** To visualise them simply run `streamlit run documentation/rollout_visualisation.py ` and visit the localhost. There you can inspect the different Variable Groups with their respective Variables and Levels.

## Resources

+ [FSDP Lighting](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)

+ Good Trainer example: https://github.com/SeanNaren/min-LLM/blob/fsdp/train.py#L235-L243

+ Interesting addition for CLI args generation: https://github.com/google/python-fire

## TODODs
- [ ] Codebase cleanup

- [ ] Hugging Face weights upload, loading and tutorial notebook.

- [ ] Make clear the data structure throughout the whole codebase. Currently we have interchanged dicts & Batch Tuples

- [x] Finetune routine implementation with LoRA and optinally VeRA DONE

- [x] Finetune dataset setup

- [x] Rollout Finetune modes: Monthly (x1), Yearly (x12)

- [x] Investigate if a (Prioritized) Buffer for Rollout Finetune is required - No need

- [x] Investigate effect of batch_size on finetuning - currently low memory usage but slow execution

- [x] Safe tensors storage

- [x] Validate distributed training strategy
