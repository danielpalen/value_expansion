import functools
from typing import Optional

import jax.lax
import haiku as hk
from brax.training import networks

from model_based_rl.types import *
from model_based_rl.utils import nonlinearity


def build_network_policy(
    obs_size: int,
    param_size: int,
    hidden_layer_sizes: Optional[Tuple[int, ...]] = (256, 256),
    weight_distribution: Optional[str] = 'uniform',
    activation: Optional[str] = 'relu',
    **kwargs
) -> networks.FeedForwardModel:
    """Creates a policy network"""

    def policy_module(input):
        net = hk.nets.MLP(
            name="policy",
            activation=nonlinearity[activation],
            output_sizes=hidden_layer_sizes + (param_size,),
            w_init=hk.initializers.VarianceScaling(
                scale=1.0, mode='fan_in', distribution=weight_distribution),
            b_init=hk.initializers.Constant(0.1)
        )
        pred = net(input)
        return pred

    policy_module = hk.without_apply_rng(hk.transform(policy_module))
    policy = networks.FeedForwardModel(
        init=lambda key: policy_module.init(key, jnp.zeros((1, obs_size))),
        apply=policy_module.apply
    )
    return policy


def build_gaussian_policy(
    policy_model,
    obs_normalizer,
    parametric_action_distribution,
    **kwargs
):
    def gaussian_policy(params, key, obs, mean=False) -> Tuple[jnp.array, jnp.array]:
        policy_params, normalizer_params = params
        norm_obs = obs_normalizer.apply(normalizer_params, obs)

        dist_params = policy_model.apply(policy_params, norm_obs)

        if mean:
            action, _ = jnp.split(dist_params, 2, axis=-1)
        else:
            action = parametric_action_distribution.sample_no_postprocessing(
                dist_params, key)

        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        norm_action = parametric_action_distribution.postprocess(action)
        return norm_action, log_prob

    return gaussian_policy


def build_gaussian_policy_fixed_noise(
    policy_model,
    obs_normalizer,
    parametric_action_distribution,
    policy_args,
    clip_noise=False,
    **kwargs
):
    def gaussian_policy_fixed_noise(params, key, obs, mean=False) -> Tuple[jnp.array, jnp.array]:
        policy_params, normalizer_params = params
        norm_obs = obs_normalizer.apply(normalizer_params, obs)

        dist_params = policy_model.apply(policy_params, norm_obs)

        action, _ = jnp.split(dist_params, 2, axis=-1)
        dist_params = jnp.concatenate(
            [action, policy_args.noise * jnp.ones_like(action)], axis=-1)  # Fixed

        if not mean:
            action_noise = policy_args.noise * \
                jax.random.normal(key, action.shape)
            if clip_noise:
                action_noise = jnp.clip(action_noise, -0.5, 0.5)
            action = action + action_noise

        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        norm_action = parametric_action_distribution.postprocess(action)

        return norm_action, log_prob

    return gaussian_policy_fixed_noise


def build_uniform_policy(
    action_size,
    **kwargs
):
    def uniform_policy(params, key, obs, min_action=-1.0, max_action=1.0) -> Tuple[jnp.array, jnp.array]:
        batch_size = obs.shape[0]
        norm_action = jax.random.uniform(
            key, (batch_size, action_size), minval=min_action, maxval=max_action)
        log_prob = jnp.log(1. / jnp.float32(max_action - min_action)
                           ) * action_size * jnp.ones(batch_size)
        return norm_action, log_prob

    return uniform_policy


build_policy_module = {
    'network': build_network_policy,
}


build_policy = {
    'gaussian': build_gaussian_policy,
    'gaussian_fixed_noise': build_gaussian_policy_fixed_noise,
    'gaussian_fixed_noise_clipped': functools.partial(build_gaussian_policy_fixed_noise, clip_noise=True),
    'uniform': build_uniform_policy,
}
