#!/bin/bash
# Usage (for iridis/slurm):
# sbatch train_tj_hard_commnet.sh [seed=0]
#
# Optional config arguments:
#     Communication: --comm_action_one --comm_mask_zero
#     Algorithms: --commnet, --ic3net, --tarcomm --ic3net (TarMAC), --gacomm

#SBATCH --ntasks=8      # Number of tasks
#SBATCH --nodes=1       # Number of nodes
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00:00 # Walltime (maximum is 60 hours for batch)

# Specific setup for my cluster and virtual environment
source activate marl37

export OMP_NUM_THREADS=1

if [ $#=1 ]; then
  seed=$1
else
  seed=1
fi

printf -v date '%(%Y-%m-%d_%H:%M:%S)T' -1

python -u run_baselines.py \
  --env_name traffic_junction \
  --nagents 20 \
  --dim 18 \
  --max_steps 80 \
  --add_rate_min 0.05 \
  --add_rate_max 0.05 \
  --difficulty hard \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 1500 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --tarcomm \
  --commnet \
  --recurrent \
  --save \
  --save_adjacency \
  --save_every 25 \
  --seed $seed \
  --load "/home/jabn1n20/CTDComm/saved/traffic_junction_hard/tar_commnet/run"$seed"/model.pt" \
#  | tee train_tj_hard_$date.log # Move to currect dir after the run.

#  num_epochs was originally 4000