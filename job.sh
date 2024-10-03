#!/bin/bash
#SBATCH --job-name=bfm-model
#SBATCH --partition=gpu_h100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1

set -e
 
 
module purge
# module load 2023 CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1 NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1 protobuf-python/4.24.0-GCCcore-12.3.0 CMake/3.26.3-GCCcore-12.3.0 Python/3.11.3-GCCcore-12.3.0
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
 
#this can be created with scripts/install_pytorch.sh
source ${venv_path}/bin/activate
 
#this is a bit hacky - exporting the paths to the python libraries
export LD_LIBRARY_PATH=${venv_path}/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=${venv_path}/lib/python3.11/site-packages/:$PYTHONPATH
 
# export DATA_PATH=/projects/0/prjs0986/wp13/dataset/
# export DATA_PATH=/projects/0/prjs0986/wp13/dataset/culturax-nl/nl/token_ids/
 
# export CHECKPOINTS_PATH=./checkpoints
 
# Export folder to log EAR verbose to
# export SLURM_EARL_VERBOSE_PATH=ear_logs
 
 
# export NCCL_DEBUG=INFO # for debugging NCCL

mlflow server --host 0.0.0.0 --port 5002 &
 
 
export TRAINING_COMMAND="python src/bfm/src/train.py"
 
srun $TRAINING_COMMAND

sleep 6000
