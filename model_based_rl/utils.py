import os
from typing import Callable, Any
import jax.numpy as jnp
import flax
import jax
import pickle
import bz2
from brax.io.file import File

from pathlib import Path

nonlinearity = {
    'elu': jax.nn.elu,
    'relu': jax.nn.relu,
    'softplus': jax.nn.softplus,
    'swish': jax.nn.swish,
    'tanh': jax.nn.tanh,
    'linear': lambda x: x,
}

# AlphaParams is the UNCONSTRAINED representation of alpha, i.e. AlphaParams \in [-inf, +inf]
# Alpha is the CONSTRAINED representation of alpha, i.e., Alpha >= 0
AlphaParams = jnp.ndarray
Alpha = jnp.ndarray


@flax.struct.dataclass
class TemperatureTransform:
    # Transformation used in the SAC alpha_loss computation.
    loss: Callable[[AlphaParams], Alpha]
    # Transformation alpha_params -> alpha
    apply: Callable[[AlphaParams], Alpha]
    # Transformation alpha -> alpha_params
    inverse: Callable[[AlphaParams], Alpha]


temperature_transforms = {
    'log_alpha': TemperatureTransform(
        loss=lambda params: params,
        apply=lambda params: jnp.exp(params),
        inverse=lambda alpha: jnp.log(alpha)
    ),
    'alpha': TemperatureTransform(
        loss=lambda params: jnp.exp(params),
        apply=lambda params: jnp.exp(params),
        inverse=lambda alpha: jnp.log(alpha)
    ),
    'softplus_alpha': TemperatureTransform(
        loss=lambda params: jax.nn.softplus(params),
        apply=lambda params: jax.nn.softplus(params),
        inverse=lambda alpha: jnp.log(jnp.exp(alpha) - 1.)
    )
}


def load(path: str) -> Any:
    with bz2.BZ2File(path, 'rb') as fin:
        buf = fin.read()

    return pickle.loads(buf)


def save(path: str, params: Any):
    """Saves parameters in Flax format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with bz2.BZ2File(path, 'wb') as fout:
        fout.write(pickle.dumps(params))

    print('SAVED to', path)


def is_slurm_job():
    """Checks whether the script is run within slurm"""
    return bool(len({k: v for k, v in os.environ.items() if 'SLURM' in k}))
