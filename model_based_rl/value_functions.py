# linear policy

import functools
from typing import Dict, Optional

import chex
import jax.lax
import numpy as np
import haiku as hk
from brax.training import networks
from ml_collections import ConfigDict

from model_based_rl.types import *
from model_based_rl.utils import nonlinearity


def build_network_value_function(
    obs_size: int,
    action_size: int,
    num_critics: int = 2,
    hidden_layer_sizes: Optional[Tuple[int, ...]] = (256, 256),
    weight_distribution: Optional[str] = 'uniform',
    activation: Optional[str] = 'relu',
    **kwargs
) -> networks.FeedForwardModel:
    """Creates a q value network"""

    def q_module(obs: jnp.ndarray, actions: jnp.ndarray):
        net = hk.nets.MLP(
            name="q_function",
            activation=nonlinearity[activation],
            output_sizes=hidden_layer_sizes + (1,),
            w_init=hk.initializers.VarianceScaling(
                scale=1.0, mode='fan_in', distribution=weight_distribution),
            b_init=hk.initializers.Constant(0.1)
        )
        return net(jnp.concatenate([obs, actions], axis=-1)).squeeze(axis=-1)

    q_module = hk.without_apply_rng(hk.transform(q_module))
    value = networks.FeedForwardModel(
        init=lambda key: jax.vmap(q_module.init, in_axes=[0, None, None])(
            jax.random.split(key, num_critics), jnp.zeros(
                (1, obs_size)), jnp.zeros((1, action_size))
        ),
        apply=jax.vmap(q_module.apply, in_axes=[0, None, None], out_axes=-1)
    )
    return value


build_value_module = {
    'network': build_network_value_function,
}
