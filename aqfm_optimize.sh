#!/bin/bash
#SBATCH --job-name=aqfm_model
#SBATCH --partition=gpu_h100
#SBATCH --time=55:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node=1

set -e

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
# export MASTER_PORT=39591
# export WORLD_SIZE=$SLURM_NTASKS
# export RANK=$SLURM_PROCID
# export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
# export LOCAL_RANK=$SLURM_LOCALID
# export NUM_NODES=$SLURM_PACK_SIZE
# export NUM_GPUS=$SLURM_GPUS_ON_NODE

venv_path=venv/ # local environment

# this can be created with scripts/install_pytorch.sh
source ${venv_path}bin/activate

# this is a bit hacky - exporting the paths to the python libraries
export LD_LIBRARY_PATH=${venv_path}/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=${venv_path}/lib/python3.11/site-packages/:$PYTHONPATH

# export DATA_PATH=/projects/0/prjs0986/wp13/dataset/
# export DATA_PATH=/projects/0/prjs0986/wp13/dataset/culturax-nl/nl/token_ids/
# export CHECKPOINTS_PATH=./checkpoints

# Export folder to log EAR verbose to
# export SLURM_EARL_VERBOSE_PATH=ear_logs

# For debugging NCCL
# export NCCL_DEBUG=INFO

mlflow server --host 0.0.0.0 --port 8082 &

export TRAINING_COMMAND="python -m test.test_bfm_alternate_version.src.train"

srun $TRAINING_COMMAND

sleep 10000
