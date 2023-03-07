from typing import Any, Callable, Mapping, Tuple
from brax.training.types import PRNGKey, Params
import jax.numpy as jnp
import optax
import flax
import brax

Action = jnp.array
Observation = jnp.array
NormObservation = jnp.array
NextObservation = jnp.array
Metrics = Mapping[str, jnp.ndarray]
DynamicsModel = Any


@flax.struct.dataclass
class Transition:
    """Contains data for one environment step."""
    s_tm1: brax.QP            # simulator state at time t-1
    o_tm1: jnp.ndarray        # observation at time t-1
    norm_o_tm1: jnp.ndarray   # normalized observation at time t-1
    a_tm1: jnp.ndarray        # action at time t-1
    log_p_tm1: jnp.ndarray    # action log prob
    s_t: brax.QP              # simulator state at time t
    o_t: jnp.ndarray          # observation at time t
    norm_o_t: jnp.ndarray     # normalized observation at time t
    r_t: jnp.ndarray          # reward
    d_t: jnp.ndarray          # discount (1-done) = NOT done flag
    truncation_t: jnp.ndarray

    # trajectory of length K. These fields are usually filled when reading an entire
    # trajectory from the replay buffer. RETRACE for example uses these fields.
    # the first entry of these fields is by definition always identical with the above fields.
    o_tm1_to_K: jnp.ndarray
    norm_o_tm1_to_K: jnp.ndarray
    a_tm1_to_K: jnp.ndarray
    o_t_to_K: jnp.ndarray
    norm_o_t_to_K: jnp.ndarray
    log_p_tm1_to_K: jnp.ndarray
    r_t_to_K: jnp.ndarray
    d_t_to_K: jnp.ndarray
    truncation_t_to_K: jnp.ndarray


@flax.struct.dataclass
class ReplayBuffer:
    """Contains data related to a replay buffer."""
    data: jnp.ndarray
    current_position: jnp.ndarray
    current_size: jnp.ndarray
    max_size: int
    empirical_delta_var: jnp.array


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    dynamics_optimizer_state: optax.OptState
    dynamics_model_params: Params
    target_policy_params: Params
    target_q_params: Params
    key: PRNGKey
    steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: Params
    # rewarder_state: Any


# The rewarder allows to change the reward of before the learner trains.
RewarderState = Any
RewarderInit = Callable[[int, PRNGKey], RewarderState]
ComputeReward = Callable[[RewarderState, Transition,
                          PRNGKey], Tuple[RewarderState, jnp.ndarray, Metrics]]
Rewarder = Tuple[RewarderInit, ComputeReward]
