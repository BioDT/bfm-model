#!/bin/bash

echo "entry point"

# all_values=(0 1 2 3 4 5 6 7 8)
# all_values=(10 11 12)
all_values=(0 1 2)

job_id_previous_batch=$(sbatch --parsable -a ${all_values[0]} create_batch.slurm)
echo "job_id_previous_batch=$job_id_previous_batch currrent_index=None"
required_jobs=$job_id_previous_batch

n_values=${#all_values[@]}

for current_index in "${!all_values[@]}"
do
  current_value=${all_values[$current_index]}
  # train with current index
  job_id_this_train=$(sbatch --parsable --dependency=afterok:$required_jobs -a $current_value train_model.slurm)
  # job_id_this_train=$job_id_this_train
  echo "current_index=$current_index current_value=$current_value"
  # if [ "$current_index" -le "$((n_values - 1))" ]
  if [ "$current_index" -ge "$((n_values - 1))" ]
  then
    echo "last iteration current_index=$current_index, current_value=$current_value"
    required_jobs=$job_id_this_train
  else
    next_value=${all_values[$((current_index + 1))]}
    echo "has next: current_index=$current_index, current_value=$current_value, next_value=$next_value"
    # generate next batch
    job_id_next_batch=$(sbatch --parsable --dependency=afterok:$required_jobs -a $next_value create_batch.slurm)
    # echo "next_value=$next_value job_id_next_batch=$job_id_next_batch"
    required_jobs=$job_id_next_batch,$job_id_this_train
  fi
  sbatch --dependency=afternotok:$required_jobs -a $current_index fail.slurm
done
