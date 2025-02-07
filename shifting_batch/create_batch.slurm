#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rome
#SBATCH --time=01:00:00

srun echo "creating batch for time index SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# TODO: call parallel_batch_create.sh with value $SLURM_ARRAY_TASK_ID
# cd ../bfm-data
# source venv/bin/activate (bfm-data)
# srun python src/dataset_creation/parallel_batch.py run-single $SLURM_ARRAY_TASK_ID
