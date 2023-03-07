#!/bin/bash
#SBATCH -J iclr23
#SBATCH -a 1-9
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:1
#SBATCH -p rtx2
#SBATCH -t 72:00:00
#SBATCH -o [TODO]
#SBATCH -e [TODO]

# Setup Env
SCRIPT_PATH=$(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
source $SCRIPT_PATH/_setup_env.sh
setup

WANDB_API_KEY="[TODO YOUR KEY]" # or wandb login
WANDB_ENTITY="[TODO YOUR ENTITY]"

python run_experiment.py \
          -env $ENV \
          -algo $ALGO \
          -model_type $MODEL \
          -H $HORIZON \
          -gpu $GPU \
          -seed $SLURM_ARRAY_TASK_ID \
          -wandb_mode "online"\
          -wandb_entity $WANDB_ENTITY\
          -wandb_project "iclr23_value_expansion"
