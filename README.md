# Diminishing Return of Value Expansion Methods in Model-Based Reinforcement Learning (ICLR 2023)

[**Palenicek, D.; Lutter, M.; Carvalho, J.; Peters, J. (2023). Diminishing Return of Value Expansion Methods in Model-Based Reinforcement Learning. International Conference on Learning Representations (ICLR).**](https://openreview.net/pdf?id=H4Ncs5jhTCu)

**Official code repository for reproducing experiments in the paper.**

If you find this code useful, please reference in your paper:
```
@inproceedings{palenicek2023value_expansion,
  author    = {Palenicek, D. and  Lutter, M. and  Carvalho, J. and  Peters, J.},
  title     = {Diminishing Return of Value Expansion Methods in Model-Based Reinforcement Learning},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

This repository provides JAX implementations of the follwing algorithms
- Soft Actor Critic (SAC) - [Haarnoja et al. 2018](https://arxiv.org/pdf/1801.01290.pdf)
- Deep Deterministic Policy Gradients (DDPG) - [Lillicrap et al. 2019](https://arxiv.org/pdf/1509.02971.pdf)
- Retrace - [Munos et al. 2016](https://arxiv.org/pdf/1606.02647.pdf)
- Model-based Value Expansion (MVE) - [Feinberg et al. 2018](https://arxiv.org/pdf/1803.00101.pdf)
- Stochastic Value Gradients (SVG) - [Heess et al. 2015](https://arxiv.org/pdf/1510.09142.pdf)

All model-based algorithms can be run with a learned neural network dynamics model and with a brax based oracle dynamics model to
eliminate model-errors for empirical analysis as done in the paper.

## Setup
1. Clone the repository
```
git clone https://github.com/danielpalen/value_expansion.git
```
2. Optionally create a [wandb](https://wandb.ai) account for experiment logging
3. Chose either Docker or Conda for setup.

**NOTE:** While it is possible to run the code on CPU only, the setup instructions assume a NVIDIA GPU. If you want to run on CPU only, then you simply need to adjust the dependencies.

### Docker
You need a working installation of Docker and the Nvidia Container Toolkit.
We provide a sample docker build file. Use this command to build the docker Image.
```
docker build -t value_expansion -f Dockerfile .
```

To run an interactive container, just run
```
docker run --rm --gpus all -it value_expansion:latest /bin/bash
```

Alternatively, you can directly pass the experiment command to the container and run it without intermediate steps
```
docker run --rm --gpus all -it value_expansion:latest python run_experiment.py -wandb_mode disabled
```
NOTE: to use wandb, you need to either pass the `WANDB_API_KEY` env variable to the container, use the `wandb docker` command or `wandb login` within the container.

### Conda
If you cannot use Docker, you can setup a working conda environment.
Make sure to have CUDA setup.

Create a conda environment and install the dependencies analoguously to how it is done in the `Dockerfile`.
```
conda create -n value_expansion python=3.7

conda activate value_expansion

cd brax
pip install -e .[develop]

cd ..
pip install -e .

wandb login
```

If you do not have the right version of CUDA on your host system installing it right in conda might be an option:
```
conda activate value_expansion
conda install -c anaconda cudatoolkit==11.8
```

## Running Experiments
After setup installation, the main script is `scripts/experiments/run_experiment.py`. The usage is as follows:
```
run_experiment.py [-h]
                         [-env {inverted_pendulum,cartpole,hopper,walker2d,halfcheetah}]
                         [-algo {SAC,SAC-RETRACE,SAC-MVE,SAC-VG,DDPG,DDPG-RETRACE,DDPG-MVE,DDPG-VG}]
                         [-H H] [-model_type {analytic,network}]
                         [-nparticles NPARTICLES] [-seed SEED] [-gpu GPU]
                         [-wandb_mode {online,disabled}]
                         [-wandb_entity WANDB_ENTITY]
                         [-wandb_project WANDB_PROJECT]

optional arguments:
  -h, --help                     show this help message and exit
  -env {inverted_pendulum,cartpole,hopper,walker2d,halfcheetah}
                                 Environment to use
  -algo {SAC,SAC-RETRACE,SAC-MVE,SAC-VG,DDPG,DDPG-RETRACE,DDPG-MVE,DDPG-VG}
                                 Algorithm to use
  -H H                           Value Expansion rollout horizon >= 0 (0 equals vanilla SAC)
  -model_type {analytic,network} Choose between oracle model (analytic) and a learned NN model
  -nparticles NPARTICLES         Number of particles per value expansion target (default 1)
  -seed SEED                     Experiment seed
  -gpu GPU                       ID of the GPU to run on. Set to -1 to run on CPU
  -wandb_mode {online,disabled}  Toggle Wandb logging
  -wandb_entity WANDB_ENTITY     WandB entity to log to
  -wandb_project WANDB_PROJECT   WandB project to log into
```

Example usage to train SAC-MVE with an oracle model and H=5 on InvertedPendulum:
```
python run_experiment.py -env inverted_pendulum -algo SAC-MVE -H 5 -model_type analytic -gpu 0
```

NOTE: In order to activate wandb logging you also need to provide the following flags: `-wandb_mode online -wandb_entity [WANDB_ENTITY]`.

### Slurm
The large experiment sweeps in the paper were performed on a [slurm cluster](https://slurm.schedmd.com/documentation.html). An example script is located in `scripts/shell/slurm_job.sh`.
The script might need adaptation to run on your cluster.

We also provide a small script `scripts/shell/submit_slurm_jobs.sh` which submits all configurations from the paper as slurm jobs using the above script `slum_job.sh` script.

## Acknowledgement
- The brax repository (https://github.com/google/brax) is forked into the subfolder `/brax` with slight modifications in the API which were required for this project. 
- Some of the code is inspired by the Brax SAC implementation.
