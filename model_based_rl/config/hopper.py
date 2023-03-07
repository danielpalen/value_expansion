from ml_collections import ConfigDict
import jax.numpy as jnp


def random_search_definition():
    definition = ConfigDict()
    definition.num_timesteps = [2_500_000,]

    definition.env = ConfigDict()
    definition.env.num_envs = [128, 256, 512]
    definition.env.discounting = [0.95, 0.97, 0.99]

    definition.sac = ConfigDict()
    definition.sac.alpha_learning_rate = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    definition.sac.alpha_transform = ['softplus_alpha', 'log_alpha', 'alpha']
    definition.sac.reward_scaling = [0.1, 1., 3.0, 5.0, 10.]
    return definition


def grid_definition():
    definition = ConfigDict()
    definition.sac = ConfigDict()
    definition.sac.alpha_learning_rate = [1.e-3, 5.e-4, 1.e-4, 5.e-5]
    definition.sac.alpha_transform = ['softplus_alpha', 'alpha', 'log_alpha']
    definition.sac.reward_scaling = [0.1, 1.]
    return definition


def termination_fn(obs: jnp.ndarray, acts: jnp.ndarray, next_obs: jnp.ndarray):
    del obs, acts

    min_z, max_z = 0.7, float('inf')
    height = next_obs[..., 0]
    is_healthy = jnp.where(height < min_z, x=0.0, y=1.0)
    is_healthy = jnp.where(height > max_z, x=0.0, y=is_healthy)
    return 1. - is_healthy
