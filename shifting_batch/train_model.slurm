#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rome
#SBATCH --time=01:00:00

srun echo "training model for time index SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
