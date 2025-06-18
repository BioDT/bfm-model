#!/bin/bash
#SBATCH --job-name=bfm_finetune
#SBATCH --partition=gpu_a100
#SBATCH --time=18:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node=4

module purge

module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

set -e

run_path=/home/atrantas/bfm-model
venv_path=${run_path}/venv # local environment

# this can be created with scripts/install_pytorch.sh
source ${venv_path}/bin/activate

# this is a bit hacky - exporting the paths to the python libraries
export LD_LIBRARY_PATH=${venv_path}/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=${venv_path}/lib/python3.11/site-packages/:$PYTHONPATH

srun python ${run_path}/bfm_model/bfm/rollout_finetuning.py
