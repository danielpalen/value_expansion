from ml_collections import ConfigDict
import jax.numpy as jnp


def termination_fn(obs: jnp.ndarray, acts: jnp.ndarray, next_obs: jnp.ndarray):
    del acts
    return jnp.where(jnp.abs(obs[..., 0]) > 1.0, 1.0, 0.0)
