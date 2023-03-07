import jax
from typing import Dict, Optional
from brax.training import networks
from model_based_rl.types import *
from model_based_rl import targets


def build_default_critic_loss(
    policy: Callable[[Tuple[Params, Params], PRNGKey, Observation], Action],
    critic_model: networks.FeedForwardModel,
    discounting: float,
    reward_scaling: float,
    **kwargs,
) -> Callable[[Params, TrainingState, jnp.ndarray, Transition, PRNGKey], Tuple[jnp.ndarray, Dict[str, Any]]]:

    def critic_loss(
            q_params: Params,
            policy_params: Params,
            normalizer_params: Params,
            target_q_params: Params,
            dynamics_params: Params,
            alpha: jnp.ndarray,
            transitions: Transition,
            key: PRNGKey,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:

        q_old_action = critic_model.apply(
            q_params, transitions.norm_o_tm1, transitions.a_tm1)
        next_action, next_log_prob = policy(
            (policy_params, normalizer_params), key, transitions.o_t)
        next_q = critic_model.apply(
            target_q_params, transitions.norm_o_t, next_action)
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.r_t * reward_scaling + transitions.d_t * discounting * next_v)
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss, {'target_q': target_q, 'q_old': q_old_action}

    return critic_loss


def build_retrace_critic_loss(
    policy: Callable[[Tuple[Params, Params], PRNGKey, Observation], Action],
    critic_model: networks.FeedForwardModel,
    discounting: float,
    reward_scaling: float,
    lambda_: float,
    policy_model: networks.FeedForwardModel,
    parametric_action_distribution,
    **kwargs,
) -> Callable[[Params, TrainingState, jnp.ndarray, Transition, PRNGKey], Tuple[jnp.ndarray, Dict[str, Any]]]:

    def retrace_loss(
            q_params: Params,
            policy_params: Params,
            normalizer_params: Params,
            target_q_params: Params,
            dynamics_params: Params,
            alpha: jnp.ndarray,
            transitions: Transition,
            key: PRNGKey,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:

        target_q = targets.retrace_target(
            policy_params=policy_params,
            normalizer_params=normalizer_params,
            target_q_params=target_q_params,
            policy=policy,
            policy_model=policy_model,
            critic_model=critic_model,
            parametric_action_distribution=parametric_action_distribution,
            alpha=alpha,
            discounting=discounting,
            reward_scaling=reward_scaling,
            lambda_=lambda_,
            transitions=transitions,
            key=key,
            stop_target_gradients=True,
        )
        target_q, target_metrics = jax.lax.stop_gradient(target_q)
        q_old_action = critic_model.apply(
            q_params, transitions.norm_o_tm1_to_K[:, :-1], transitions.a_tm1_to_K[:, :-1])

        valid_mask = target_metrics['valid_mask']

        # Select first time step
        target_q = target_q[:, 0]
        q_old_action = q_old_action[:, 0]

        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss, {
            'target_q': target_q,
            'q_old': q_old_action,
            'traj_length': jnp.sum(valid_mask, axis=1),
            **target_metrics,
        }

    return retrace_loss


def build_mve_critic_loss_retrace(
    policy: Callable[[Tuple[Params, Params], PRNGKey, Observation], Action],
    critic_model: networks.FeedForwardModel,
    dynamics_model: DynamicsModel,
    discounting: float,
    reward_scaling: float,
    horizon: int,
    lambda_: float,
    **kwargs,
) -> Callable[[Params, TrainingState, jnp.ndarray, Transition, PRNGKey], Tuple[jnp.ndarray, Dict[str, Any]]]:
    assert horizon >= 0

    retrace = build_retrace_critic_loss(
        policy, critic_model, discounting, reward_scaling, lambda_,
        kwargs['policy_model'], kwargs['parametric_action_distribution'],
    )

    def mve_critic_loss(
            q_params: Params,
            policy_params: Params,
            normalizer_params: Params,
            target_q_params: Params,
            dynamics_params: Params,
            alpha: jnp.ndarray,
            transitions: Transition,
            key: PRNGKey,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:

        rollout_key, retrace_key = jax.random.split(key)

        transitions = dynamics_model.rollout_transitions(
            rollout_key, transitions,
            (policy_params, normalizer_params),
            (dynamics_params, normalizer_params),
            policy, horizon,
            start_at_t=True
        )

        loss, info = retrace(q_params, policy_params, normalizer_params, target_q_params,
                             dynamics_params, alpha, transitions, retrace_key)
        return loss, info

    return mve_critic_loss


build = {
    'default': build_default_critic_loss,
    'retrace': build_retrace_critic_loss,
    'mve': build_mve_critic_loss_retrace,
}
