from ml_collections import ConfigDict
import jax.numpy as jnp


def termination_fn(obs: jnp.ndarray, acts: jnp.ndarray, next_obs: jnp.ndarray):
    del obs, acts
    return jnp.zeros_like(next_obs[..., 0])
