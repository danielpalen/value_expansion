from typing import Tuple
import chex
import jax
import scipy
import numpy as np
import jax.numpy as jnp
from functools import partial

import brax
from brax.envs import env


def coordinates2index(coord: jnp.ndarray, dimension=Tuple[int, int]):
  chex.assert_shape(coord, (None, 2))
  chex.assert_type(coord, int)

  idx = coord[:, 0] * dimension[0] + coord[:, 1]
  return idx


def index2coordinates(idx: jnp.ndarray, dimension=Tuple[int, int]):
  chex.assert_shape(idx, (None,))
  chex.assert_type(idx, int)

  coord_1 = jnp.mod(idx, dimension[0])
  coord_0 = (idx - coord_1) // dimension[0]

  return jnp.stack([coord_0, coord_1], axis=-1)

def index2action(idx: jnp.array):
  chex.assert_shape(idx, (None,))
  chex.assert_type(idx, int)

  action = jnp.array([
    [1, 0],  # Down
    [0, 1],  # Right
    [-1, 0],  # Up
    [0, -1],  # Left
  ])
  return action[idx, :]


class Gridworld(env.Env):
  def __init__(self, n=8, dt=1., **kwargs):
    _SYSTEM_CONFIG = f"dt: {dt}"
    super().__init__(_SYSTEM_CONFIG, **kwargs)
    self.n = n
    self.dimension = jnp.array([self.n, self.n])

    # Transforms to map state index to coordinates
    self.idx2coord = partial(index2coordinates, dimension=self.dimension)
    self.coord2idx = partial(coordinates2index, dimension=self.dimension)

    # Transform to map action index to coordinates
    self.idx2action = index2action

    # If the environment is deterministic the action are performed with a probability of 1. Otherwise the actions have
    # a non-zero probability to execute a different action comparable to frozen lake.
    self.deterministic = True

    # With easy exploration, the agent is initialized anywhere on the gridworld. Otherwise the agent is only initialized
    # in the state [0, 0]
    self.easy_exploration = True

    # Definition of the environment
    self.init_states = jnp.array([8, 8]) if self.easy_exploration else jnp.array([0, 0])
    self.death_states = self.coord2idx(jnp.array([[90, 90], ])) # [5, 7], [1, 5]
    self.goal_states = self.coord2idx(jnp.array([[7, 7]]))
    self.death_rwd = -100
    self.goal_rwd = 100
    self.step_rwd = -1

    # Termination Function
    terminal_states = jnp.concatenate([self.death_states, self.goal_states])
    self.termination_fn = partial(termination_fn, terminal_states=terminal_states)

  def reset(self, rng: jax.random.PRNGKey) -> env.State:
    x0 = jax.random.randint(rng, shape=(1, 2), minval=jnp.array([0, 0]), maxval=self.init_states, dtype=int)
    idx0 = self.coord2idx(x0)
    return env.State(qp=None, obs=idx0, reward=jnp.float32(0.), done=jnp.float32(0))

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    return state.replace(
      obs=self.dyn_fn(state, action),
      reward=self.rwd_fn(state, action),
      done=self.termination_fn(state.obs, None, None)
    )

  def dyn_fn(self, state: env.State, action: jnp.ndarray) -> jnp.ndarray:
    coord = self.idx2coord(state.obs)
    next_coord = jnp.clip(coord + self.idx2action(action), a_min=jnp.zeros(2, dtype=int), a_max=self.dimension-1)
    return self.coord2idx(next_coord)

  def rwd_fn(self, state: env.State, action: jnp.ndarray) -> jnp.ndarray:
    distance_to_death = jnp.abs(state.obs[:, jnp.newaxis] - self.death_states[jnp.newaxis, :])
    is_alive = jnp.clip(jnp.min(distance_to_death, axis=-1), a_min=0, a_max=1).squeeze()
    is_dead = 1 - is_alive

    distance_to_goal = jnp.abs(state.obs[:, jnp.newaxis] - self.goal_states[jnp.newaxis, :])
    not_goal = jnp.clip(jnp.min(distance_to_goal, axis=-1), a_min=0, a_max=1).squeeze()
    in_goal = 1 - not_goal

    rwd = self.step_rwd * not_goal * is_alive + self.death_rwd * is_dead + self.goal_rwd * in_goal
    return rwd.astype(jnp.float32)

  @property
  def action_size(self):
    return 4

  @property
  def number_of_states(self):
    return jnp.prod(self.dimension).item()


def termination_fn(obs: jnp.ndarray, acts: jnp.ndarray, next_obs: jnp.ndarray, terminal_states: jnp.array):
  chex.assert_rank([terminal_states, obs], 1)

  distance = jnp.abs(obs[:, jnp.newaxis] - terminal_states[jnp.newaxis, :])
  not_terminated = jnp.clip(jnp.min(distance, axis=-1), a_min=0, a_max=1).squeeze()
  done = 1 - not_terminated
  return done.astype(jnp.float32)


if __name__ == "__main__":
  # Testing the basic functionality of the gridworld:
  dimension = (8, 8)
  coord = jnp.array([[0, 0], [1, 1], [7, 6], [3, 2], [7, 7]])
  idx = coordinates2index(coord, dimension)
  recreated_coord = index2coordinates(idx, dimension)
  assert jnp.allclose(coord, recreated_coord)

  idx = jnp.arange(64, dtype=int)
  coord = index2coordinates(idx, dimension)
  state = env.State(qp=None, obs=idx, reward=jnp.zeros(idx.shape), done=jnp.zeros(idx.shape))
  action = jax.random.randint(jax.random.PRNGKey(0), shape=idx.shape, minval=0, maxval=4)

  env = Gridworld()
  next_state = env.step(state, action)

  # Visualize the Index:
  map = jnp.zeros(dimension, dtype=int)
  map = map.at[coord[:, 0], coord[:, 1]].set(idx)
  print(f"Map:\n{map}\n")

  # Visualize the Reward:
  rwd = env.rwd_fn(state, action)
  rwd_map = jnp.zeros(dimension, dtype=int)
  rwd_map = rwd_map.at[coord[:, 0], coord[:, 1]].set(rwd)
  print(f"Reward Map:\n{rwd_map}\n")

  # Visualize Termination Function:
  done = env.termination_fn(state.obs, action, state.obs)
  done_map = jnp.zeros(dimension, dtype=int)
  done_map = done_map.at[coord[:, 0], coord[:, 1]].set(done)
  print(f"Done Map:\n{done_map}\n")

  # Visualize the Dynamics:
  action2string = {0: "down", 1: "right", 2: "up", 3: "left"}

  for i in range(64):
    print(f"{i:02d}: {state.obs[i]:02d} + {action2string[action[i].item()]:>5} -> {next_state.obs[i]:02d}")