#!/bin/bash
#SBATCH -J docker_iclr23
#SBATCH -a 1-9
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:1
#SBATCH -p rtx2
#SBATCH -t 72:00:00
#SBATCH -o [TODO]
#SBATCH -e [TODO]

# TODO: for logging experiments you need to provide you WANDB info here.
WANDB_API_KEY="[TODO YOUR KEY]"  # or 'wandb login' and 'wandb docker' commands
WANDB_ENTITY="[TODO YOUR ENTITY]"

# Start the rootless Docker daemon on the compute node
source ias-rootless-dockerd-start

# Run the Docker image
docker run --rm --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    value_expansion:latest \
    python run_experiment.py \
          -env $ENV \
          -algo $ALGO \
          -model_type $MODEL \
          -H $HORIZON \
          -gpu $GPU \
          -seed $SLURM_ARRAY_TASK_ID \
          -wandb_mode "online"\
          -wandb_entity $WANDB_ENTITY \
          -wandb_project "iclr23_value_expansion"
