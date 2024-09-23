#!/bin/bash
# Usage:
# sbatch train_multiple_experiments.sh foo_bar.sh [num_runs]
# Positional arguments:
#     experiment script (e.g. train_tj_easy_commnet.sh)
# Optional argument:
#     number of training runs to do

#SBATCH --ntasks=1      # Number of tasks
#SBATCH --nodes=1       # Number of nodes
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00 # Walltime (maximum is 60 hours for batch)

if [ $# -eq 1 ]; then
    num_runs=5
else
    num_runs=$2
fi

# This is here in case we've already got a run from testing
if [ $num_runs -eq 4 ]; then
    start=2
    num_runs=5
else
    start=1
fi

for ((i = $start ; i < $num_runs+1 ; i++ )); do
    sbatch $1 $i
    sleep 60
done