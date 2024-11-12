#!/bin/bash
# Usage (for iridis/slurm, from CTDComm/baselines):
# sbatch dec_scripts/train_dpp_5v1_dec_tarmac.sh [seed=1]
#
# Optional config arguments:
#     Communication: --comm_action_one --comm_mask_zero
#     Algorithms: --commnet, --ic3net, --tarcomm --ic3net (TarMAC), --gacomm

#SBATCH --ntasks=8      # Number of tasks
#SBATCH --nodes=1       # Number of nodes
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00:00 # Walltime (maximum is 60 hours for batch)

# Specific setup for my cluster and virtual environment
# making sure there aren't any other conda envs active to interfere
module load conda
# source deactivate
# source deactivate
source activate commpy
# conda init
# conda deactivate
# conda deactivate
# conda activate commpy

export OMP_NUM_THREADS=1

if [ $# -eq 1 ]; then
  seed=$1
else
  seed=1
fi

printf -v date '%(%Y-%m-%d_%H:%M)T' -1 

# Hyperparameters follow TarMAC where known
python -u run_baselines.py \
  --env_name dec_predator_prey \
  --nagents 10 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --num_epochs 250 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --alpha 0.99 \
  --gamma 0.99 \
  --entr 0.01 \
  --lrate 0.0007 \
  --dec_tarmac \
  --message_augment \
  --ic3net \
  --comm_passes 2 \
  --recurrent \
  --save \
  --save_every 25 \
  --seed $seed \
  --env_seed $seed \

  # Total epochs = 1000

  
# Things to test
  # --message_augment \ # default v_augment