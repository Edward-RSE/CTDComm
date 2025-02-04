#!/bin/bash

#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=00:10:00

source $HOME/modules/ctdcomm.lmod
source $HOME/codes/CTDComm/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

if [ $# -eq 1 ]; then
  seed=$1
else
  seed=1
fi

printf -v date '%(%Y-%m-%d_%H:%M:%S)T' -1

python_exe=pyinstrument
# python_exe=python

$python_exe run_baselines.py \
  --env_name dec_predator_prey \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 1 \
  --num_epochs 2 \
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

# Parameters from TarMAC paper.
  # --alpha 0.99 \
  # --gamma 0.99 \
  # --entr 0.01 \
  # --lrate 0.0007 \