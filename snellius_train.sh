#!/bin/bash
#SBATCH --job-name=bfm_model
#SBATCH --partition=gpu_h100
#SBATCH --time=3:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node=4
#SBATCH --reservation=gpt-nl

module purge

module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

set -e

run_path=/gpfs/home2/damian/projects/bfm-model
venv_path=${run_path}/venv # local environment

# this can be created with scripts/install_pytorch.sh
source ${venv_path}/bin/activate

# this is a bit hacky - exporting the paths to the python libraries
export LD_LIBRARY_PATH=${venv_path}/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=${venv_path}/lib/python3.11/site-packages/:$PYTHONPATH

srun python ${run_path}/bfm_model/bfm/train_lighting.py
