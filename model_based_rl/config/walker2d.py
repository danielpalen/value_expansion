from ml_collections import ConfigDict
import jax.numpy as jnp


def random_search_definition():
    definition = ConfigDict()
    definition.num_timesteps = [2_500_000,]

    definition.env = ConfigDict()
    definition.env.num_envs = [2048,]
    definition.env.num_eval_envs = [128,]
    definition.env.discounting = [0.95, 0.97, 0.98, 0.99]

    definition.sac = ConfigDict()
    definition.sac.log_alpha = [0.0, -1.0, -2.0]
    definition.sac.reward_scaling = [0.1, 1., 3.0, 5.0, 10.]
    return definition


def grid_definition():
    raise NotImplementedError
    # definition = ConfigDict()
    # definition.env = ConfigDict()
    # definition.env.discounting = [0.99, 0.99, 0.98, 0.99]
    #
    # definition.sac = ConfigDict()
    # definition.sac.log_alpha = [-2.0, -2.0, 0.0, 0.0]
    # definition.sac.reward_scaling = [0.1, 1., 1.0, 10.]
    # return definition


def termination_fn(obs: jnp.ndarray, acts: jnp.ndarray, next_obs: jnp.ndarray):
    del obs, acts

    min_z, max_z = 0.7, 2.0
    height = next_obs[..., 0]
    is_healthy = jnp.where(height < min_z, x=0.0, y=1.0)
    is_healthy = jnp.where(height > max_z, x=0.0, y=is_healthy)
    return 1. - is_healthy
