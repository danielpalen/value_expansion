import os
import wandb
import argparse
from datetime import datetime
from ml_collections import ConfigDict
from model_based_rl import hyperparameters
from model_based_rl import config
from model_based_rl import utils

ENV_CHOICES = ['inverted_pendulum', 'cartpole',
               'hopper', 'walker2d', 'halfcheetah']
ALGO_CHOICES = [
    'SAC-RETRACE',
    'SAC-MVE',      # Critic expansion
    'SAC-VG',       # Actor expansion
    'DDPG-RETRACE',
    'DDPG-MVE',     # Critic expansion
    'DDPG-VG'       # Actor expansion
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-env", type=str, required=False, default='inverted_pendulum', help="Environment",
                        choices=ENV_CHOICES)

    parser.add_argument("-algo", type=str, required=False,
                        default='SAC-MVE', help="Algorithm to use", choices=ALGO_CHOICES)
    parser.add_argument("-H", type=int, required=False, default=0,
                        help="MVE rollout horizon >= 0 (0 equals vanilla SAC)")
    parser.add_argument("-model_type", type=str, required=False, default='analytic',
                        help="Choose between oracle model (analytic) and a learned NN model",
                        choices=['analytic', 'network'])
    parser.add_argument("-nparticles", type=int, required=False, default=1,
                        help="Number of particles per value expansion target (default 1)")
    parser.add_argument("-seed", type=int, required=False,
                        default=1, help="Experiment seed")

    parser.add_argument("-gpu", type=int, required=False, default=0,
                        help="ID of the GPU to run on. Set to -1 to run on CPU")
    parser.add_argument("-wandb_mode", type=str, required=False, default='online', choices=['online', 'disabled'],
                        help="Toggle Wandb logging")
    parser.add_argument("-wandb_entity", type=str,
                        required=False, help="WandB entity to log to")
    parser.add_argument("-wandb_project", type=str, default='value_expansion_iclr23',
                        required=False, help="WandB project to log into")

    args = parser.parse_args()
    seed, env_name, algo = args.seed, args.env, args.algo
    assert algo in ALGO_CHOICES

    if args.wandb_mode == 'online':
        assert args.wandb_entity and args.wandb_project, \
            'When wandb_mode==online, You must specify the wandb_entity and wandb_project'

    # CUDA_VISIBLE_DEVICES must be set before importing jax
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu if args.gpu >= 0 else ''}"

    import jax
    import brax
    from brax import envs
    from model_based_rl import soft_actor_critic as sac

    # Default Hyperparameter:
    hyper = ConfigDict()
    hyper.start_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    hyper.num_timesteps = 2_500_000
    hyper.log_frequency = 100
    hyper.env_name = env_name
    hyper.algorithm = algo
    hyper.seed = seed

    hyper.env = ConfigDict()
    hyper.env.num_envs = 256
    hyper.env.num_eval_envs = 128
    hyper.env.discounting = 0.95
    hyper.env.action_repeat = 1
    hyper.env.episode_length = 1000

    hyper.sac = ConfigDict()
    hyper.sac.alpha = 1.0
    hyper.sac.alpha_learning_rate = 5e-5
    hyper.sac.alpha_transform = 'alpha'
    hyper.sac.grad_updates_per_step = 1

    hyper.sac.reward_scaling = 1.
    hyper.sac.normalize_observations = True
    hyper.sac.batch_size = 256
    hyper.sac.learning_rate = 3e-4
    hyper.sac.actor_learning_rate = hyper.sac.learning_rate
    hyper.sac.critic_learning_rate = hyper.sac.learning_rate
    hyper.sac.tau_pi = 1.0                 # Averaging of the Target network weights
    hyper.sac.tau_q = 0.005                # Averaging of the Target network weights
    # Must be multiple of Environments, i.e. base 2
    hyper.sac.min_replay_size = 65_536
    # Must be multiple of Environments, i.e. base 2
    hyper.sac.max_replay_size = 1_048_576

    hyper.sac.policy_build_fn = 'network'
    hyper.sac.policy_fn = 'gaussian'
    hyper.sac.eval_policy = 'gaussian'
    hyper.sac.mean_eval = False

    hyper.sac.critic_build_fn = 'network'
    hyper.sac.param_action_distribution_fn = 'tanhNormal'
    hyper.sac.policy_args = ConfigDict()
    hyper.sac.policy_args.noise = 0.1

    hyper.sac.network = ConfigDict()
    hyper.sac.network.weight_distribution = 'truncated_normal'
    hyper.sac.network.hidden_layer_sizes = (256, 256)
    hyper.sac.network.activation = 'relu'

    hyper.dynamics_model = ConfigDict()
    hyper.dynamics_model.model_type = args.model_type

    hyper.dynamics_model.network = ConfigDict()
    hyper.dynamics_model.network.weight_distribution = 'truncated_normal'
    hyper.dynamics_model.network.learning_rate = 3e-4
    hyper.dynamics_model.network.activation = 'relu'
    hyper.dynamics_model.network.threshold = 2.0
    hyper.dynamics_model.network.hidden_layer_sizes = (256, 256, 256, 256)
    hyper.dynamics_model.network.ensemble_size = 5
    hyper.dynamics_model.network.logvar_limits = (-10., 0.5)
    hyper.dynamics_model.network.min_updates = 1
    hyper.dynamics_model.network.num_updates = 10
    hyper.dynamics_model.network.batch_size = 256
    hyper.dynamics_model.network.init_epochs = 200
    hyper.dynamics_model.network.n_epochs = 1
    hyper.dynamics_model.network.deterministic = False
    hyper.dynamics_model.network.logvar_learned = True

    hyper.algo_config = ConfigDict()
    hyper.algo_config.name = hyper.algorithm
    hyper.algo_config.train_dynamics_model = False

    hyper.algo_config.actor_loss_config = ConfigDict()
    hyper.algo_config.actor_loss_config.loss = 'default'
    hyper.algo_config.actor_loss_config.policy_fn = 'stochastic'

    hyper.algo_config.critic_loss_config = ConfigDict()
    hyper.algo_config.critic_loss_config.loss = 'default'
    hyper.algo_config.critic_loss_config.policy_fn = 'stochastic'
    hyper.algo_config.critic_loss_config.td_k_trick = None
    hyper.algo_config.critic_loss_config.retrace_k = 0
    hyper.algo_config.critic_loss_config.lambda_ = 1.

    hyper.algo_config.dyna = ConfigDict()
    hyper.algo_config.dyna.samples = False

    # Switch to DDPG
    if algo.split('-')[0] == 'DDPG':
        hyper.sac.alpha = 0.0
        hyper.sac.alpha_learning_rate = 0.0
        hyper.algo_config.actor_loss_config.policy_fn = 'mean'
        hyper.algo_config.critic_loss_config.policy_fn = 'mean'
        hyper.sac.policy_fn = 'gaussian_fixed_noise'
        hyper.sac.tau_pi = hyper.sac.tau_q
        hyper.sac.mean_eval = True
        hyper.sac.policy_args = ConfigDict()
        hyper.sac.policy_args.noise = 0.1

    # Setup expansion method: RETRACE, MVE or VG.
    if len(algo.split('-')) > 1:
        if algo.split('-')[1] == 'RETRACE':
            hyper.algo_config.critic_loss_config.loss = 'retrace'
            hyper.algo_config.critic_loss_config.retrace_k = args.H + 2

        elif algo.split('-')[1] == 'MVE':
            hyper.algo_config.train_dynamics_model = True
            hyper.algo_config.critic_loss_config.loss = 'mve'
            hyper.algo_config.critic_loss_config.horizon = args.H

        elif algo.split('-')[1] == 'VG':
            hyper.algo_config.train_dynamics_model = True
            hyper.algo_config.actor_loss_config.loss = 'value_gradient'
            hyper.algo_config.actor_loss_config.horizon = args.H

    # Setup environment specific parameters.
    if env_name == "inverted_pendulum":
        hyper.num_timesteps = 10_000
        hyper.log_frequency = 100
        hyper.env.num_envs = 1
        hyper.env.action_repeat = 4
        hyper.sac.min_replay_size = 512

    elif env_name == "cartpole":
        hyper.num_timesteps = 100_000
        hyper.log_frequency = 100
        hyper.env.num_envs = 1
        hyper.env.action_repeat = 2
        hyper.env.discounting = 0.99
        hyper.sac.min_replay_size = 512

    elif env_name == "hopper":
        hyper.num_timesteps = 1_000_000
        hyper.log_frequency = 200
        hyper.env.num_envs = 128
        hyper.env.action_repeat = 2
        hyper.env.discounting = 0.99

    elif env_name == "walker2d":
        hyper.num_timesteps = 1_000_000
        hyper.log_frequency = 200
        hyper.env.num_envs = 128
        hyper.env.action_repeat = 2
        hyper.env.discounting = 0.99

    elif env_name == "halfcheetah":
        hyper.num_timesteps = 1_000_000
        hyper.log_frequency = 200
        hyper.env.num_envs = 128
        hyper.sac.batch_size = 512

    else:
        raise ValueError(f"Environment '{env_name}'does not exist!")

    hyper.hash = hyperparameters.compute_hash(hyper)
    hyper.lock()

    is_slurm_job = utils.is_slurm_job()

    # Set WandB config:
    name = f"{hyper.algorithm} {hyper.env_name.replace('_', ' ').capitalize()} Seed={hyper.seed:02d}"
    group = f"{hyper.algorithm}_{hyper.env_name}_H={args.H}_model={hyper.dynamics_model.model_type}"
    tags = [hyper.env_name, hyper.algorithm, 'ICLR_23_value_expansion']
    log_path = None

    with wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=name, group=group, tags=tags,
        config=hyper.to_dict(),
        settings=wandb.Settings(start_method="fork") if is_slurm_job else None,
        mode=args.wandb_mode
    ) as wandb_run:

        try:
            if log_path:
                hyper_path = os.path.join(log_path, f"checkpoints/hyper.pkl")
                utils.save(hyper_path, hyper.to_dict())
                wandb.save(hyper_path)

            if is_slurm_job:
                wandb_run.summary['SLURM_JOB_ID'] = os.environ.get(
                    'SLURM_JOBID')

            print(f"\nEnvironment Setup:")
            print(f"  Environment = {hyper.env_name}")
            print(f"  Group Name  = {group}")
            print(f"         Seed = {hyper.seed}")
            print(f"          GPU = {jax.devices()[0].device_kind}")

            # Start SAC Training
            sac.train(
                env_name=hyper.env_name,
                num_envs=hyper.env.num_envs,
                num_eval_envs=hyper.env.num_eval_envs,
                discounting=hyper.env.discounting,
                action_repeat=hyper.env.action_repeat,
                episode_length=hyper.env.episode_length,
                seed=hyper.seed,
                num_timesteps=hyper.num_timesteps,
                log_frequency=hyper.log_frequency,
                policy_build_fn=hyper.sac.policy_build_fn,
                policy_fn=hyper.sac.policy_fn,
                policy_args=hyper.sac.policy_args,
                mean_eval=hyper.sac.mean_eval,
                critic_build_fn=hyper.sac.critic_build_fn,
                reward_scaling=hyper.sac.reward_scaling,
                normalize_observations=hyper.sac.normalize_observations,
                batch_size=hyper.sac.batch_size,
                tau_pi=hyper.sac.tau_pi,
                tau_q=hyper.sac.tau_q,
                alpha=hyper.sac.alpha,
                alpha_transform=hyper.sac.alpha_transform,
                alpha_learning_rate=hyper.sac.alpha_learning_rate,
                actor_learning_rate=hyper.sac.actor_learning_rate,
                critic_learning_rate=hyper.sac.critic_learning_rate,
                min_replay_size=hyper.sac.min_replay_size,
                max_replay_size=hyper.sac.max_replay_size,
                grad_updates_per_step=hyper.sac.grad_updates_per_step,
                sac_network_config=hyper.sac.network,
                param_action_distribution_fn=hyper.sac.param_action_distribution_fn,
                model_type=hyper.dynamics_model.model_type,
                dynamics_model_config=hyper.dynamics_model.network,
                algo_config=hyper.algo_config,
                termination_fn=config.termination_fn[hyper.env_name],
                checkpoint_logdir=log_path,
                is_slurm_job=is_slurm_job,
            )

        finally:

            if is_slurm_job:
                from subprocess import check_output

                stdout_path = check_output(
                    "scontrol show jobid -dd $SLURM_JOB_ID | awk -F= '/StdOut=/{print $2}'", shell=True)
                stderr_path = check_output(
                    "scontrol show jobid -dd $SLURM_JOB_ID | awk -F= '/StdErr=/{print $2}'", shell=True)
                stdout_path = stdout_path.decode('utf-8').replace('\n', '')
                stderr_path = stderr_path.decode('utf-8').replace('\n', '')

                wandb.save(stdout_path)
                wandb.save(stderr_path)
