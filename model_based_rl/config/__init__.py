import jax

from typing import NamedTuple, Callable, Any
from model_based_rl.config import cartpole
from model_based_rl.config import halfcheetah
from model_based_rl.config import hopper
from model_based_rl.config import pendulum
from model_based_rl.config import inverted_pendulum
from model_based_rl.config import walker2d


class HyperparameterSweep(NamedTuple):
    random_search: Callable[[], Any]
    grid_search: Callable[[], Any]


sweep_definition = {
    'hopper': HyperparameterSweep(
        random_search=hopper.random_search_definition,
        grid_search=hopper.grid_definition),

    'walker2d': HyperparameterSweep(
        random_search=walker2d.random_search_definition,
        grid_search=walker2d.grid_definition)
}


termination_fn = {
    'cartpole': cartpole.termination_fn,
    'halfcheetah': halfcheetah.termination_fn,
    'hopper': hopper.termination_fn,
    'pendulum': pendulum.termination_fn,
    'inverted_pendulum': inverted_pendulum.termination_fn,
    'walker2d': walker2d.termination_fn,
}
