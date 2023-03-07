import jax.numpy as jnp


def termination_fn(obs: jnp.ndarray, acts: jnp.ndarray, next_obs: jnp.ndarray):
    del obs, acts

    cart_height = .1
    pendulum_length = .3  # joint offset

    if next_obs.shape[-1] == 4:
        # Observations = [x, theta, x_dot, theta_dot]
        theta = next_obs[..., 1]
        cos_theta = jnp.cos(theta)

    elif next_obs.shape[-1] == 5:
        # Observations = [x, cos(theta), sin(theta), x_dot, theta_dot]
        cos_theta = next_obs[..., 1]

    else:
        # Observations = The default google/brax observations.
        theta = next_obs[..., 5]
        cos_theta = jnp.cos(theta)

    z_pendulum = cos_theta * pendulum_length + cart_height
    done = jnp.where(z_pendulum > .2, jnp.float32(0), jnp.float32(1))
    return done
