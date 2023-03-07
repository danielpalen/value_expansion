import functools
from typing import Dict, Optional
import jax
import chex

from brax.training import networks
from model_based_rl.types import *
import model_based_rl.rlax as rlax


def retrace_target(
    policy_params: Params,
    normalizer_params: Params,
    target_q_params: Params,
    policy: Callable[[Tuple[Params, Params], PRNGKey, Observation], Action],
    policy_model: networks.FeedForwardModel,
    critic_model: networks.FeedForwardModel,
    parametric_action_distribution,
    alpha: jnp.ndarray,
    discounting: float,
    reward_scaling: float,
    lambda_: float,
    transitions: Transition,
    key: PRNGKey,
    stop_target_gradients: False,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:

    norm_o_t_to_K = transitions.norm_o_t_to_K[:, :-1]
    o_t_to_K = transitions.o_t_to_K[:, :-1]
    a_t_to_K = transitions.a_tm1_to_K[:, 1:]
    r_t = transitions.r_t_to_K[:, :-1] * reward_scaling
    d_t = transitions.d_t_to_K[:, :-1]
    truncation_t = transitions.truncation_t_to_K[:, :-1]

    # q_t: Q-values under π of actions executed by μ at times [1, ..., K - 1].
    q_t = jnp.min(jax.vmap(critic_model.apply, in_axes=[None, 1, 1], out_axes=1)(
        target_q_params, transitions.norm_o_t_to_K[:, :-1], transitions.a_tm1_to_K[:, 1:]), axis=-1)

    # v_t: Values under π at times [1, ..., K].
    a_pi, log_p = policy((policy_params, normalizer_params), key, o_t_to_K)
    q_t_pi = jnp.min(critic_model.apply(
        target_q_params, norm_o_t_to_K, a_pi), axis=-1)
    v_t = q_t_pi - alpha * log_p

    # discount_t: discounts at times [1, ..., K].
    not_done_tm1 = jnp.concatenate(
        [jnp.ones((d_t.shape[0], 1)), jax.lax.stop_gradient(d_t)], axis=1)
    not_done_tm1 = (1 - jnp.clip(jnp.cumsum(1 - not_done_tm1,
                    axis=1), a_min=jnp.float32(0), a_max=jnp.float32(1)))
    valid_mask = not_done_tm1[:, :-1]
    not_done_tm1 = not_done_tm1[:, 1:]
    # not_done_tm1 = jax.lax.stop_gradient(not_done_tm1)
    discount_t = discounting * not_done_tm1

    # c_t: weights at times [1, ..., K - 1].
    log_p_t_to_K = parametric_action_distribution.log_prob(
        policy_model.apply(
            policy_params, transitions.norm_o_tm1_to_K[:, 1:]),  # dist
        parametric_action_distribution.inverse_postprocess(
            # clip since arctanh can give NaNs.
            jnp.clip(transitions.a_tm1_to_K[:, 1:],
                     a_min=-1. + 1.e-9, a_max=1. - 1.e-9)
        )  # unnormalize action
    )
    # (pi/behaviour pi) in log space
    log_rhos = log_p_t_to_K - transitions.log_p_tm1_to_K[:, 1:]
    c_t = jnp.minimum(1.0, jnp.exp(log_rhos)) * lambda_ * (1. - truncation_t)

    target_q = jax.vmap(functools.partial(rlax.general_off_policy_returns_from_q_and_v))(
        q_t=q_t[:, :-1],
        v_t=v_t,
        r_t=r_t,
        discount_t=discount_t,
        c_t=jax.lax.stop_gradient(c_t[:, :-1]),
    )
    target_q = jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(
        jnp.array(target_q)), jnp.array(target_q))

    return target_q, {
        'c_t': c_t,
        'q_t': q_t,
        'a_pi': a_pi,
        'log_p': log_p,
        'q_t_pi': q_t_pi,
        'v_t': v_t,
        'discount_t': discount_t,
        'target_q': target_q,
        'valid_mask': valid_mask,
    }
