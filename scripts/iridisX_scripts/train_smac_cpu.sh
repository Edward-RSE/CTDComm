#!/bin/bash

#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=01:00:00

source .venv/bin/activate
export PYTHONUNBUFFERED=1

if [ $# -eq 1 ]; then
  seed=$1
else
  seed=1
fi

printf -v date '%(%Y-%m-%d_%H:%M:%S)T' -1


python -u run_baselines.py \
  --env_name starcraft2 \
  --map_name 3m \
  --max_steps 40 \
  --nprocesses 1 \
  --num_epochs 100 \
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
  --lrate 0.0007

