#!/bin/bash

# This script submits slurm jobs to run all of the experiments for the paper.

ENVS=("inverted_pendulum" "cartpole" "hopper" "walker2d" "halfcheetah")
ALGOS=("SAC-MVE" "SAC-VG" "SAC-RETRACE" "DDPG-MVE" "DDPG-VG" "DDPG-RETRACE")
MODEL_TYPES=("analytic" "network")
HORIZONS=(1 3 5 10 20 30)
GPU=0

for env in ${ENVS[@]}; do

  # H=0 just reduces to SAC, so we just need to run it once.
  ENV=$env ALGO='SAC-MVE' MODEL='analytic' HORIZON=0 GPU=$GPU sbatch slurm_job.sh

  # Run other configurations
  for algo in ${ALGOS[@]}; do
    for model in ${MODEL_TYPES[@]}; do
      for horizon in ${HORIZONS[@]}; do
        if [[ $algo == *RETRACE ]] && [[ $model == "network" ]]; then
          # Since retrace does not use a model we can skip one of the iterations.
          break
        fi

        ENV=$env ALGO=$algo MODEL=$model HORIZON=$horizon GPU=$GPU sbatch slurm_job.sh

      done ;
    done
  done
done
