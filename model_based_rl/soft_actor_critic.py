import os
import jax
import chex
import time
import wandb
import functools
import haiku as hk
from pathlib import Path
from typing import Dict, Optional
from ml_collections import ConfigDict

from brax import envs
from brax.io import model
from brax.training import distribution
from brax.training import env
from brax.training import networks
from brax.training import normalization

from model_based_rl import dynamics_models
from model_based_rl.replay_buffer import build_replay_buffer_functions
from model_based_rl.types import *
from model_based_rl.types import *
from model_based_rl import evaluation
from model_based_rl.utils import nonlinearity, temperature_transforms
from model_based_rl import actor_losses
from model_based_rl import critic_losses
from model_based_rl.policies import build_policy_module, build_policy
# from model_based_rl.plotting import plot_dynamics, stats
from model_based_rl import value_functions
from model_based_rl import utils

_default_sac_network_config = ConfigDict()
_default_sac_network_config.activation = 'relu'
_default_sac_network_config.weight_distribution = 'truncated_normal'
_default_sac_network_config.hidden_layer_sizes = (256, 256)
_default_sac_network_config.lock()

_default_dynamics_network_config = ConfigDict()
_default_dynamics_network_config.activation = 'relu'
_default_dynamics_network_config.weight_distribution = 'truncated_normal'
_default_dynamics_network_config.hidden_layer_sizes = (256, 256)
_default_dynamics_network_config.logvar_limits = (-10., 0.5)
_default_dynamics_network_config.ensemble_size = 5
_default_dynamics_network_config.learning_rate = 3e-4
_default_dynamics_network_config.num_updates = 16
_default_dynamics_network_config.batch_size = 32
_default_dynamics_network_config.lock()

_default_algo_config = ConfigDict()
_default_algo_config.name = 'SAC'
_default_algo_config.dyna_samples = False
_default_algo_config.actor_loss = 'default'
_default_algo_config.critic_loss = 'default'
_default_algo_config.lock()


def train(
    env_name: str,
    num_timesteps: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    log_frequency: int = 10000,
    policy_build_fn: str = 'network',
    policy_fn: str = 'gaussian',
    mean_eval: bool = False,
    policy_args: ConfigDict = ConfigDict(),
    critic_build_fn: str = 'network',
    normalize_observations: bool = False,
    reward_scaling: float = 1.,
    # Averaging of the Target network weights
    tau_pi: float = 1.0,
    # Averaging of the Target network weights
    tau_q: float = 0.005,
    alpha: float = 1.0,                         # Initial alpha / temperature
    alpha_transform: str = 'log_alpha',         # Representation of the Alpha
    alpha_learning_rate: float = 3.e-4,         # Learning rate of Alpha
    actor_learning_rate: float = 1e-4,
    critic_learning_rate: float = 1e-4,
    min_replay_size: int = 8192,
    max_replay_size: int = 1048576,
    grad_updates_per_step: float = 1,
    sac_network_config: Optional[ConfigDict] = _default_sac_network_config,
    param_action_distribution_fn: str = 'tanhNormal',
    model_type: Optional[str] = 'network',
    algo_config: Optional[ConfigDict] = _default_algo_config,
    dynamics_model_config: Optional[ConfigDict] = _default_dynamics_network_config,
    termination_fn: Optional[Callable[[Observation,
                                       Action, NextObservation], jnp.ndarray]] = None,
    checkpoint_logdir: Optional[Path] = None,
    is_slurm_job: bool = False,
    legacy_spring: bool = False,
):
    """SAC training."""
    print(f"\nSAC Setup:")

    t = time.time()
    assert min_replay_size % num_envs == 0
    assert max_replay_size % min_replay_size == 0

    key = jax.random.PRNGKey(seed)
    (
        key_local,    # -> Initial key of the training state
        key_eval,     # -> Key for the evaluations
        key_env,      # -> Key for the initial states of the train environment.
        key_eval_env,  # -> Key for the initial states of the eval environment.
        key_models,   # -> Init key for the networks
        key_rewarder  # -> Init key for the rewarder
    ) = jax.random.split(key, 6)

    num_updates = int(num_envs * grad_updates_per_step)
    eval_length = episode_length // action_repeat
    num_total_sac_epochs = num_timesteps // num_envs
    num_sac_epochs = max(num_total_sac_epochs // log_frequency, 1)

    core_env = envs.create(
        env_name=env_name,
        action_repeat=action_repeat,
        batch_size=num_envs,
        episode_length=episode_length,
        deterministic_reset=False,
        legacy_spring=legacy_spring,
    )

    step_fn = jax.jit(core_env.step)
    reset_fn = jax.jit(core_env.reset)
    first_state = reset_fn(key_env)

    core_eval_env = envs.create(
        env_name=env_name,
        action_repeat=action_repeat,
        batch_size=num_eval_envs,
        episode_length=episode_length,
        deterministic_reset=True,
        legacy_spring=legacy_spring
    )

    eval_first_state, eval_step_fn = env.wrap_for_eval(
        core_eval_env, key_eval_env)
    _, obs_size = eval_first_state.core.obs.shape
    action_size = core_env.action_size

    norm_params, obs_normalizer = normalization.create_observation_normalizer(
        obs_size, normalize_observations)

    parametric_action_distribution = {
        'tanhNormal': distribution.NormalTanhDistribution,
        'normal': distribution.ParamNormalDistribution
    }[param_action_distribution_fn](event_size=core_env.action_size)

    policy_model = build_policy_module[policy_build_fn](
        obs_size, parametric_action_distribution.param_size, action_size=action_size,
        weight_distribution=sac_network_config.weight_distribution,
        hidden_layer_sizes=sac_network_config.hidden_layer_sizes,
        activation=sac_network_config.activation,
        fixed_noise=policy_args.noise if policy_fn == 'gaussian_fixed_noise' else None
    )

    policy = build_policy[policy_fn](
        policy_model=policy_model, obs_normalizer=obs_normalizer,
        parametric_action_distribution=parametric_action_distribution,
        policy_args=policy_args
    )
    mean_policy = jax.jit(functools.partial(policy, mean=True))
    policy = jax.jit(policy)

    uniform_policy = jax.jit(build_policy['uniform'](
        action_size=core_env.action_size, policy_model=policy_model, obs_normalizer=obs_normalizer,
        parametric_action_distribution=parametric_action_distribution,
    ))

    value_model = value_functions.build_value_module[critic_build_fn](
        obs_size, core_env.action_size, weight_distribution=sac_network_config.weight_distribution,
        hidden_layer_sizes=sac_network_config.hidden_layer_sizes, activation=sac_network_config.activation,
    )

    key_policy, key_q, key_dynamics_model = jax.random.split(key_models, 3)
    policy_params = policy_model.init(key_policy)
    q_params = value_model.init(key_q)

    target_entropy = -0.5 * core_env.action_size
    alpha_transform = temperature_transforms[alpha_transform]
    alpha_params = jnp.asarray(
        alpha_transform.inverse(alpha), dtype=jnp.float32)
    alpha_optimizer = optax.adam(learning_rate=alpha_learning_rate)
    alpha_optimizer_state = alpha_optimizer.init(alpha_params)

    policy_optimizer = optax.adam(learning_rate=actor_learning_rate)
    policy_optimizer_state = policy_optimizer.init(policy_params)

    q_optimizer = optax.adam(learning_rate=critic_learning_rate)
    q_optimizer_state = q_optimizer.init(q_params)

    build_replay_buffer, update_replay_buffer, sample_replay_buffer = build_replay_buffer_functions(
        max_replay_size, num_envs, obs_normalizer,
        calculate_empirical_delta_var=dynamics_model_config.deterministic,
        trajectory_window=algo_config.critic_loss_config.retrace_k
    )

    dynamics_model, dynamics_model_params, dynamics_optimizer_state = None, None, None
    if algo_config.train_dynamics_model:

        # Create & Initialize the Model:
        dynamics_model = dynamics_models.available_models[model_type](
            env_name=env_name,
            obs_size=obs_size,
            acts_size=action_size,
            normalizer=obs_normalizer,
            action_repeat=action_repeat,
            termination_fn=termination_fn,
            sample_replay_buffer_fn=functools.partial(
                sample_replay_buffer, num_envs=num_envs),
            ensemble_size=dynamics_model_config.ensemble_size,
            learning_rate=dynamics_model_config.learning_rate,
            batch_size=dynamics_model_config.batch_size,
            n_epochs=dynamics_model_config.n_epochs,
            deterministic=dynamics_model_config.deterministic,
            logvar_learned=dynamics_model_config.logvar_learned,
            num_updates=dynamics_model_config.num_updates * num_envs,
            min_updates=dynamics_model_config.min_updates * num_envs,
            threshold=dynamics_model_config.threshold,
            activation=dynamics_model_config.activation,
            weight_distribution=dynamics_model_config.weight_distribution,
            hidden_layer_sizes=dynamics_model_config.hidden_layer_sizes,
            logvar_limits=dynamics_model_config.logvar_limits,
        )

        dynamics_model_params, dynamics_optimizer_state = dynamics_model.init(
            key_dynamics_model)

    run_eval = jax.jit(functools.partial(
        evaluation.run,
        policy_fn=mean_policy if mean_eval else policy,
        step_fn=eval_step_fn,
        length=eval_length)
    )

    def alpha_loss(alpha_params: jnp.ndarray,
                   policy_params: Params,
                   transitions: Transition,
                   key: PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = policy_model.apply(policy_params, transitions.norm_o_tm1)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        log_alpha_loss = alpha_transform.loss(
            alpha_params) * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(log_alpha_loss), -jnp.mean(log_prob)

    critic_loss = critic_losses.build[algo_config.critic_loss_config.loss](
        discounting=discounting,
        reward_scaling=reward_scaling,
        policy={
            'mean': mean_policy,
            'stochastic': policy,
            'gaussian_fixed_noise_clipped': build_policy['gaussian_fixed_noise_clipped'](
                policy_model=policy_model, obs_normalizer=obs_normalizer,
                parametric_action_distribution=parametric_action_distribution,
                policy_args=policy_args
            )
        }[algo_config.critic_loss_config.policy_fn],
        policy_model=policy_model,
        parametric_action_distribution=parametric_action_distribution,
        critic_model=value_model,
        dynamics_model=dynamics_model,
        **algo_config.critic_loss_config,
    )

    actor_loss = actor_losses.build[algo_config.actor_loss_config.loss](
        reward_scaling=reward_scaling,
        discounting=discounting,
        policy={
            'mean': mean_policy,
            'stochastic': policy,
            'gaussian_fixed_noise_clipped': build_policy['gaussian_fixed_noise_clipped'](
                policy_model=policy_model, obs_normalizer=obs_normalizer,
                parametric_action_distribution=parametric_action_distribution,
                policy_args=policy_args
            )
        }[algo_config.actor_loss_config.policy_fn],
        policy_model=policy_model,
        parametric_action_distribution=parametric_action_distribution,
        critic=value_model,
        dynamics_model=dynamics_model,
        **algo_config.actor_loss_config,
    )

    alpha_grad = jax.jit(jax.value_and_grad(alpha_loss, has_aux=True))
    critic_grad = jax.jit(jax.value_and_grad(critic_loss, has_aux=True))
    actor_grad = jax.jit(jax.value_and_grad(actor_loss, has_aux=True))

    @jax.jit
    def update_step(state: TrainingState, transitions: Transition) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
        (key, key_alpha, key_critic, key_actor,
         key_rewarder) = jax.random.split(state.key, 5)

        (alpha_loss, entropy), alpha_grads = alpha_grad(
            state.alpha_params, state.policy_params,
            transitions, key_alpha
        )

        alpha = alpha_transform.apply(state.alpha_params)

        (critic_loss, critic_metrics), critic_grads = critic_grad(
            state.q_params, state.target_policy_params, state.normalizer_params, state.target_q_params,
            state.dynamics_model_params, alpha, transitions, key_critic
        )

        (actor_loss, _), actor_grads = actor_grad(
            state.policy_params, state.q_params, state.normalizer_params, state.dynamics_model_params, alpha,
            transitions, key_actor
        )

        # Update Policy:
        policy_params_update, policy_opt_state = policy_optimizer.update(
            actor_grads, state.policy_optimizer_state)
        policy_params = optax.apply_updates(
            state.policy_params, policy_params_update)

        # Update Q-Function:
        q_params_update, q_opt_state = q_optimizer.update(
            critic_grads, state.q_optimizer_state)
        q_params = optax.apply_updates(state.q_params, q_params_update)

        # Update Alpha:
        alpha_params_update, alpha_opt_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(
            state.alpha_params, alpha_params_update)

        # Update the target network weights:
        def polyak_average(t, x, y): return jax.tree_map(
            lambda x, y: x * (1 - t) + y * t, x, y)
        new_target_policy_params = polyak_average(
            tau_pi, state.target_policy_params, policy_params)
        new_target_q_params = polyak_average(
            tau_q, state.target_q_params, q_params)

        flat_alpha_grads = jnp.concatenate(jax.tree_map(
            lambda x: x.reshape(-1), jax.tree_flatten(alpha_grads)[0]))
        flat_actor_grads = jnp.concatenate(jax.tree_map(
            lambda x: x.reshape(-1), jax.tree_flatten(actor_grads)[0]))
        flat_critic_grads = jnp.concatenate(jax.tree_map(
            lambda x: x.reshape(-1), jax.tree_flatten(critic_grads)[0]))

        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': entropy,
            'grads/alpha_mean': jnp.mean(flat_alpha_grads),
            'grads/actor_mean': jnp.mean(flat_actor_grads),
            'grads/critic_mean': jnp.mean(flat_critic_grads),
            'grads/alpha_abs_mean': jnp.mean(jnp.abs(flat_alpha_grads)),
            'grads/actor_abs_mean': jnp.mean(jnp.abs(flat_actor_grads)),
            'grads/critic_abs_mean': jnp.mean(jnp.abs(flat_critic_grads)),
            'grads/alpha_std': jnp.std(flat_alpha_grads),
            'grads/actor_std': jnp.std(flat_actor_grads),
            'grads/critic_std': jnp.std(flat_critic_grads),
            **critic_metrics,
        }

        new_state = TrainingState(
            policy_optimizer_state=policy_opt_state,
            policy_params=policy_params,
            q_optimizer_state=q_opt_state,
            q_params=q_params,
            dynamics_optimizer_state=state.dynamics_optimizer_state,
            dynamics_model_params=state.dynamics_model_params,
            target_policy_params=new_target_policy_params,
            target_q_params=new_target_q_params,
            key=key,
            steps=state.steps + 1,
            alpha_optimizer_state=alpha_opt_state,
            alpha_params=alpha_params,
            normalizer_params=state.normalizer_params
        )

        return new_state, metrics

    def collect_data(training_state: TrainingState, state: brax.QP, policy: Callable) \
            -> Tuple[TrainingState, brax.QP, Transition]:

        key, key_sample = jax.random.split(training_state.key)
        actions, log_probs = policy(
            (training_state.policy_params, training_state.normalizer_params), key_sample, state.obs)
        nstate = step_fn(state, actions)

        normalizer_params = obs_normalizer.update(
            training_state.normalizer_params, state.obs)
        training_state = training_state.replace(
            key=key, normalizer_params=normalizer_params)

        # For the RL updates, the done flag should only be true **if** the done was triggered by an exception and **not**
        # by the episode wrapper terminating because of reaching the maximum steps. To revert done's by the episode wrapper
        # the done flag is determined by: 'nstate.done AND NOT nstate.info[truncation]'.
        done = jnp.logical_and(
            nstate.done, jnp.logical_not(nstate.info['truncation']))

        # The next state and observation are read from the info dict as the nstate.qp / obs do not necessarily contain the
        # next state. If the environment was done, the autoreset wrapper writes the next initial state in to the nstate. To
        # always have access to the **real** next state, the next state is saved to the info dict.
        transition_data = Transition(
            s_tm1=state.qp,
            o_tm1=state.obs,
            norm_o_tm1=None,
            a_tm1=actions,
            log_p_tm1=log_probs,
            s_t=nstate.info['qp'],
            o_t=nstate.info['obs'],
            norm_o_t=None,
            r_t=nstate.reward,
            d_t=1.0 - done,
            # Whether the trajectory was terminated by the episode wrapper.
            truncation_t=nstate.info['truncation'],
            o_tm1_to_K=None,
            norm_o_tm1_to_K=None,
            a_tm1_to_K=None,
            o_t_to_K=None,
            norm_o_t_to_K=None,
            log_p_tm1_to_K=None,
            r_t_to_K=None,
            d_t_to_K=None,
            truncation_t_to_K=None,
        )

        return training_state, nstate, transition_data

    def collect_and_update_buffer(
        training_state: TrainingState,
        state: brax.QP,
        replay_buffer: ReplayBuffer,
        policy: Callable
    ) -> Tuple[TrainingState, brax.QP, ReplayBuffer]:
        training_state, state, newdata = collect_data(
            training_state, state, policy)
        new_replay_buffer = update_replay_buffer(replay_buffer, newdata)
        return training_state, state, new_replay_buffer

    def init_replay_buffer(training_state: TrainingState, state: brax.QP, replay_buffer: ReplayBuffer, policy: Callable) \
            -> Tuple[TrainingState, brax.QP, ReplayBuffer]:
        (training_state, state, replay_buffer), _ = jax.lax.scan(
            lambda a, b: ((collect_and_update_buffer(*a, policy=policy)), ()),
            (training_state, state, replay_buffer),
            (),
            length=min_replay_size // num_envs)

        return training_state, state, replay_buffer

    def run_one_sac_epoch(carry, unused_t):
        training_state, state, replay_buffer = carry
        metrics = {}

        # Sample the real environment:
        training_state, state, replay_buffer = collect_and_update_buffer(
            training_state, state, replay_buffer, policy)

        # Sample replay Buffer
        training_state, transitions = sample_replay_buffer(
            training_state, replay_buffer, batch_size, num_updates, num_envs)

        # Train dynamics model
        if algo_config.train_dynamics_model:
            # Train the dynamics model:
            training_state, model_train_metrics = dynamics_model.train(
                training_state, replay_buffer)
            metrics.update(model_train_metrics)

        # Training Loop
        training_state, train_metrics = jax.lax.scan(
            update_step, training_state, transitions)
        metrics.update(train_metrics)

        metrics['buffer_current_size'] = replay_buffer.current_size
        metrics['buffer_current_position'] = replay_buffer.current_position
        return (training_state, state, replay_buffer), metrics

    def run_sac_training(training_state, state, replay_buffer):
        (training_state, state, replay_buffer), metrics = jax.lax.scan(
            run_one_sac_epoch,
            (training_state, state, replay_buffer),
            (),
            length=num_sac_epochs
        )
        metrics = jax.tree_map(jnp.mean, metrics)
        return training_state, state, replay_buffer, metrics

    # Initial Training State
    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        dynamics_model_params=dynamics_model_params,
        dynamics_optimizer_state=dynamics_optimizer_state,
        target_policy_params=policy_params,
        target_q_params=q_params,
        key=key_local,
        steps=jnp.zeros((1,)),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=alpha_params,
        normalizer_params=norm_params,
    )

    # Initial Entropy:
    dist_params = policy_model.apply(policy_params, first_state.obs)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    initial_entropy = - \
        jnp.mean(parametric_action_distribution.log_prob(dist_params, action))

    training_walltime_iter = 0
    training_walltime = 0
    training_metrics = {'entropy': initial_entropy}
    state = first_state
    eval_walltime = 0
    iter = 0
    sps = 0

    t_setup = time.time() - t
    print(f"   Setup Time = {t_setup:.1f}s")

    # Initialize the Replay Buffer
    t = time.time()
    _, _, transition_init_data = collect_data(
        training_state, first_state, policy)
    replay_buffer = build_replay_buffer(transition_init_data, obs_size)
    training_state, state, replay_buffer = init_replay_buffer(
        training_state, state, replay_buffer, uniform_policy)
    replay_steps = int(training_state.normalizer_params[0])
    replay_walltime = time.time() - t
    print(f"Replay Memory = {replay_walltime:.1f}s")

    if checkpoint_logdir:
        # Save entire training state
        checkpoint = training_state, replay_buffer
        chkpt_path = os.path.join(
            checkpoint_logdir, f"checkpoints/checkpoint_{iter:03d}.pkl")
        utils.save(os.path.join(checkpoint_logdir,
                   f"checkpoints/checkpoint_{iter:03d}.pkl"), checkpoint)
        wandb.save(chkpt_path)

    t = time.time()
    if algo_config.train_dynamics_model:
        print(f"Dynamics Model Pre-training = {replay_walltime:.1f}s")
        training_state, _ = jax.jit(dynamics_model.train)(
            training_state, replay_buffer, dynamics_model_config.init_epochs)
    print(f"\nStart Training:")

    # Main training and evaluation loop
    while True:
        iter = iter + 1
        current_step = int(training_state.normalizer_params[0])
        t = time.time()

        eval_state, key_eval, eval_traj = run_eval(eval_first_state, key_eval, training_state.policy_params,
                                                   training_state.normalizer_params)

        eval_state.completed_episodes.block_until_ready()
        eval_walltime_iter = time.time() - t
        eval_walltime += eval_walltime_iter
        eval_sps = (episode_length // action_repeat *
                    eval_first_state.core.reward.shape[0] / (time.time() - t))
        avg_episode_length = (
            eval_state.completed_episodes_steps / eval_state.completed_episodes)

        metrics = dict(
            dict({
                f'eval/episode_{name}': value / eval_state.completed_episodes
                for name, value in eval_state.completed_episodes_metrics.items()
            }),
            **dict({
                f'training/{name}': value
                # if 'grads' not in name
                for name, value in training_metrics.items()
            }),
            **dict({
                'eval/completed_episodes': eval_state.completed_episodes,
                'eval/avg_episode_length': avg_episode_length,
                'speed/sps': sps,
                'speed/eval_sps': eval_sps,
                'speed/training_total_walltime': training_walltime,
                'speed/training_walltime': training_walltime_iter,
                'speed/eval_total_walltime': eval_walltime,
                'speed/eval_walltime': eval_walltime_iter,
                'x-axis/grad_updates': training_state.steps[0],
                'x-axis/env_step': int(current_step/num_envs),
                'x-axis/norm_env_step': int((current_step - replay_steps)/num_envs),
                'x-axis/total_env_step': current_step,
            }),
        )

        print(
            f"({iter:04d}) - # Grad = {metrics['x-axis/grad_updates']:.1e} / # Env = {current_step:.1e}  => "
            f"Reward = {metrics['eval/episode_reward']:+07.1f} / {metrics['eval/completed_episodes']:04.0f} - "
            f"Entropy = {metrics['training/entropy']:+0.2f} - "
            f"SPS = {metrics['speed/sps']:.2e} / {metrics['speed/eval_sps']:.2e} - "
            f"Wallclock = {metrics['speed/training_walltime']:02.0f}s/{metrics['speed/training_total_walltime']:03.0f}s / "
            f"{metrics['speed/eval_walltime']:02.0f}s/{metrics['speed/eval_total_walltime']:03.0f}s"
        )

        wandb.log(metrics, step=current_step)

        if checkpoint_logdir and (current_step >= num_timesteps or jnp.mod(iter, 200) == 0):
            # Save training state
            checkpoint = training_state, replay_buffer
            chkpt_path = os.path.join(
                checkpoint_logdir, f"checkpoints/checkpoint_{iter:03d}.pkl")
            utils.save(os.path.join(checkpoint_logdir,
                       f"checkpoints/checkpoint_{iter:03d}.pkl"), checkpoint)
            wandb.save(chkpt_path)

        if current_step >= num_timesteps:
            break

        # Reset autoreset environment states when they are depleted:
        if jnp.max(state.info['idx']) > core_env.num_states:
            key, reset_key = jax.random.split(training_state.key)
            training_state = training_state.replace(key=key)
            reset_info = reset_fn(reset_key).info
            state.info.update(
                idx=reset_info['idx'], first_qp=reset_info['first_qp'], first_obs=reset_info['first_obs'])

        # Optimization
        t = time.time()
        training_state, state, replay_buffer, training_metrics = run_sac_training(
            training_state, state, replay_buffer)
        jax.tree_map(lambda x: x.block_until_ready(), training_metrics)
        sps = (
            (training_state.normalizer_params[0] - current_step) / (time.time() - t))
        training_walltime_iter = time.time() - t
        training_walltime += training_walltime_iter

    print(f"Max Resets per Environment = {jnp.max(state.info['idx'])} - "
          f"Steps per Env = {current_step/num_envs:.0f} / {episode_length // action_repeat:.0f} \n\n")
