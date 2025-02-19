#!/bin/bash
#SBATCH --job-name=bfm_model
#SBATCH --partition=gpu_h100
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node=1

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0

set -e

venv_path=venv/ # local environment

# this can be created with scripts/install_pytorch.sh
source ${venv_path}bin/activate

# module purge
# module load 2023 Python/3.11.3-GCCcore-12.3.0
# pip install -e .

# this is a bit hacky - exporting the paths to the python libraries
export LD_LIBRARY_PATH=${venv_path}/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=${venv_path}/lib/python3.11/site-packages/:$PYTHONPATH

srun python src/bfm/src/train_lighting.py