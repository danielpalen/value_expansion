import jax
import scipy
import numpy as np
import jax.numpy as jnp

import brax
from brax.envs import env


class LinearSystem(env.Env):
  def __init__(self, n=1, dt=1./50., **kwargs):
    _SYSTEM_CONFIG = f"dt: {dt}"
    super().__init__(_SYSTEM_CONFIG, **kwargs)

    self.n = n
    self.dt = dt

    # Define linear dynamics
    A_cont = jnp.eye(n) * 0.05
    B_cont = jnp.eye(n) * 0.1 / dt

    self.A, self.B = jnp.eye(n) + dt * A_cont, dt * B_cont

    # Define cost function
    self.Q = jnp.eye(n)
    self.R = jnp.eye(n)

    # Optimal LQR policy for plotting.
    # PQ = DeviceArray([[11.639289 ,  1.062866 ], [ 1.062866 ,  1.1061804]], dtype=float32)
    # P  = array([[10.61804239]])
    # K  = 0.9608432650566101

    self.PQ, self.P, self.K = build_lqr_policy(self.A, self.B, self.Q, self.R)

  def reset(self, rng: jax.random.PRNGKey) -> env.State:
    x0 = jax.random.uniform(rng, shape=(self.n,), minval=-4.5, maxval=+4.5)
    cost = jnp.float32(0.) # TODO: how to set initial cost?
    return env.State(qp=None, obs=x0, reward=-cost, done=jnp.float32(0))

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    return state.replace(
      obs=self.dyn_fn(state, action),
      reward=-self.cost_fn(state, action),
      done=termination_fn(state.obs, action, None)
    )

  def dyn_fn(self, state: env.State, action: jnp.ndarray) -> jnp.ndarray:
    return self.A @ state.obs + self.B @ action

  def cost_fn(self, state: env.State, action: jnp.ndarray) -> jnp.ndarray:
    # (state cost) + (action cost)
    return (state.obs.T @ self.Q @ state.obs) + (action.T @ self.R @ action)

  @property
  def action_size(self):
    return self.n


def build_lqr_policy(A, B, Q, R):
  # Compute optimal gains for a linear System and a quadratic cost function
  # x' = A x + B u
  # J  = \sum^{\infty}_{0} x^T Q x + u^T R u

  # Compute P by solving the continuous algebraic Riccati Equation.
  P_mat = scipy.linalg.solve_discrete_are(A, B, Q, R)

  # Compute the optimal gains:
  BTPB = jnp.matmul(B.transpose(), jnp.matmul(P_mat, B))
  BTPA = jnp.matmul(B.transpose(), jnp.matmul(P_mat, A))
  # K_mat = jnp.matmul(jnp.linalg.pinv(R + BTPB), BTPA)
  K_mat = jnp.matmul(np.linalg.pinv(R + BTPB), BTPA)  # TODO: doing this in np for now since the JAX op fails on the cluster.

  # Compute Q-function:
  ATPA = jnp.matmul(A.transpose(), jnp.matmul(P_mat, A))
  ATPB = jnp.matmul(A.transpose(), jnp.matmul(P_mat, B))
  BTPA = ATPB.transpose()

  Q_mat = jnp.concatenate(
      [jnp.concatenate([Q + ATPA, ATPB], axis=-1),
       jnp.concatenate([BTPA, R + BTPB], axis=-1)],
       axis=0)

  return Q_mat, P_mat, K_mat

def termination_fn(obs: jnp.ndarray, acts: jnp.ndarray, next_obs: jnp.ndarray):
  # 2D Circle
  # return jnp.float32(jnp.sqrt(jnp.square(obs[..., 0]) + jnp.square(obs[..., 1])) > 10)

  # # Box
  return jnp.clip(jnp.sum(jnp.float32(jnp.concatenate([obs < -10, 10 < obs], axis=-1)), axis=-1), a_min=0, a_max=1)
