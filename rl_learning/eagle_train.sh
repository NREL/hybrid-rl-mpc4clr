#!/bin/bash
#SBATCH --account=drl4dsr
#SBATCH --time=4:00:00
#SBATCH --job-name=train_rl_agent
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36

source ~/.bashrc

module purge
module load conda
module load xpressmp
conda activate rl_mpc_env

unset LD_PRELOAD

python ray_train.py --num-cpus 35