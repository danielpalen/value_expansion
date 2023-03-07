import functools

import chex
import jax.lax
from brax.training import networks
from ml_collections import ConfigDict

from model_based_rl.types import *
from model_based_rl import targets
import model_based_rl.rlax as rlax


def build_default_actor_loss(
        policy: Callable[[Tuple[Params, Params], PRNGKey, Observation], Action],
        critic: networks.FeedForwardModel,
        **kwargs
) -> Callable[[Params, Params, jnp.ndarray, Transition, PRNGKey], Tuple[jnp.ndarray, dict]]:

    def actor_loss(
        policy_params: Params,
        q_params: Params,
        normalizer_params: Params,
        dynamics_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> Tuple[jnp.ndarray, dict]:
        batch_size = transitions.o_tm1.shape[0]

        # The policy normalizes implicitly ...
        action, log_prob = policy(
            (policy_params, normalizer_params), key, transitions.o_tm1)
        q_action = critic.apply(q_params, transitions.norm_o_tm1, action)
        min_q = jnp.min(q_action, axis=-1)

        chex.assert_shape(action, (batch_size, None))
        chex.assert_shape(q_action, (batch_size, 2))
        chex.assert_shape(log_prob, (batch_size, ))
        chex.assert_shape(min_q, (batch_size,))

        loss = -1. * (min_q - alpha * log_prob)
        chex.assert_shape(loss, (batch_size,))
        return jnp.mean(loss), {}

    return actor_loss


def build_retrace_value_gradient_actor_loss(
        policy: Callable[[Tuple[Params, Params], PRNGKey, Observation], Action],
        policy_model,
        parametric_action_distribution,
        critic: networks.FeedForwardModel,
        dynamics_model: DynamicsModel,
        reward_scaling: float,
        discounting: float,
        horizon: int,
        lambda_: float = 1.0,
        **kwargs
) -> Callable[[Params, Params, jnp.ndarray, Transition, PRNGKey], Tuple[jnp.ndarray, dict]]:

    if horizon == 0:
        return build_default_actor_loss(policy=policy, critic=critic)

    def actor_loss(
        policy_params: Params,
        q_params: Params,
        normalizer_params: Params,
        dynamics_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> Tuple[jnp.ndarray, dict]:
        batch_size = transitions.o_tm1.shape[0]

        rollout_key, retrace_key = jax.random.split(key)

        transitions = dynamics_model.rollout_transitions(
            rollout_key, transitions,
            (policy_params, normalizer_params),
            (dynamics_params, normalizer_params),
            policy, horizon, start_at_t=False
        )

        target_q, _ = targets.retrace_target(
            policy_params=policy_params,
            normalizer_params=normalizer_params,
            target_q_params=q_params,
            policy=policy,
            policy_model=policy_model,
            critic_model=critic,
            parametric_action_distribution=parametric_action_distribution,
            alpha=alpha,
            discounting=discounting,
            reward_scaling=reward_scaling,
            lambda_=lambda_,
            transitions=transitions,
            key=retrace_key,
            stop_target_gradients=False,
            # **kwargs,
        )

        value_t0 = target_q[:, 0] - alpha * transitions.log_p_tm1_to_K[:, 0]

        # amos et al. divide by horizon:
        # https://github.com/facebookresearch/svg/blob/eff39ca93abdd4fd07afe81bd826b5805dd4c028/svg/agent.py#L233
        loss = - jnp.mean(value_t0) / (horizon + 1)

        return loss, {'transitions': transitions}

    return actor_loss


build = {
    'default': build_default_actor_loss,
    'value_gradient': build_retrace_value_gradient_actor_loss,
}
