from typing import Dict, Optional, Tuple, Any

import chex
import jax
from brax.training import env


def run(
        state,
        key,
        policy_params,
        normalizer_params,
        policy_fn,
        step_fn,
        length
) -> Tuple[env.EvalEnvState, chex.PRNGKey, Any]:
    next_key, eval_key = jax.random.split(key)

    def do_one_step(carry, key_sample):
        state, params = carry
        actions, _ = policy_fn(params, key_sample, state.core.obs)
        nstate = step_fn(state, actions)
        return (nstate, params), {'state': state.core.qp, 'observation': state.core.obs, 'reward': state.core.reward}

    (state, _), traj = jax.lax.scan(
        do_one_step,
        (state, (policy_params, normalizer_params)),
        jax.random.split(eval_key, length))

    return state, next_key, traj
