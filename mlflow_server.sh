#!/bin/bash
#SBATCH --job-name=mlflow_server
#SBATCH --partition=thin
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -e

# defaults for host and port
HOST=${1:-0.0.0.0}
PORT=${2:-8082}

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0

venv_path=venv/

source ${venv_path}bin/activate

export LD_LIBRARY_PATH=${venv_path}/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=${venv_path}/lib/python3.11/site-packages/:$PYTHONPATH

mlflow server --host $HOST --port $PORT

sleep 7200
