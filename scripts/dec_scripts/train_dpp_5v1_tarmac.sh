#!/bin/bash
# Usage (for iridis/slurm):
# sbatch train_pp_medium_commnet.sh [seed=0]
#
# Optional config arguments:
#     Communication: --comm_action_one --comm_mask_zero
#     Algorithms: --commnet, --ic3net, --tarcomm --ic3net (TarMAC), --gacomm

#SBATCH --ntasks=8      # Number of tasks
#SBATCH --nodes=1       # Number of nodes
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00:00 # Walltime (maximum is 60 hours for batch)

# Specific setup for my cluster and virtual environment
module load conda
source activate commpy_37 #commpy for Iridis 5
# conda init
# conda activate commpy
#marl37

export OMP_NUM_THREADS=1

if [ $#=1 ]; then
  seed=$1
else
  seed=1
fi

printf -v date '%(%Y-%m-%d_%H:%M:%S)T' -1 

python -u run_baselines.py \
  --env_name dec_predator_prey \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --ic3net \
  --tarcomm \
  --recurrent \
  --comm_passes 2 \
  --save \
  --save_every 25 \
  --seed $seed \
  --env_seed $seed \
  --alpha 0.99 \
  --gamma 0.99 \
  --entr 0.01 \
  --lrate 0.0007 \


# Parameters from TarMAC paper.
  # --alpha 0.99 \
  # --gamma 0.99 \
  # --entr 0.01 \
  # --lrate 0.0007 \