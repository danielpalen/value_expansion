# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An inverted pendulum environment."""
import jax.numpy as jnp
import brax
from brax import jumpy as jp
from brax.envs import env
from brax.envs import rewards


class Pendulum(env.Env):
  """Trains an inverted pendulum to remain stationary."""

  def __init__(self, feature_transform=True, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config, **kwargs)

    # With Feature Transform -> obs = [cart pos, cos(angle), sin(angle), cart vel, angle vel]
    # Without Feature Transform -> obs = [cart pos, angle, cart vel, angle vel]
    self.feature_transform = feature_transform

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    # qpos = jp.array([jp.pi]) + jp.random_uniform(rng1, (self.sys.num_joint_dof,), -.01, .01)
    qpos = jp.random_uniform(rng1, (self.sys.num_joint_dof,), -jp.pi, jp.pi)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.01, .01)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done = self._get_rwd(obs, jp.zeros(1)), jp.float32(0.0)
    metrics = {}
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    reward = self._get_rwd(obs, action)

    return state.replace(qp=qp, obs=obs, reward=reward)

  @property
  def action_size(self):
    return 1

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe cos/sin joint angle and velocities."""
    # some pre-processing to pull joint angles and velocities
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    if self.feature_transform:
      qpos = [jp.cos(joint_angle), jp.sin(joint_angle)]
    else:
      qpos = [joint_angle]

    qvel = [joint_vel]
    return jp.concatenate(qpos + qvel)

  def _get_rwd(self, obs, action):
    upright = (obs[1] + 1) / 2
    centered = rewards.tolerance(obs[0], margin=2)
    centered = (1 + centered) / 2

    small_control = rewards.tolerance(action, margin=1, value_at_margin=0, sigmoid='quadratic')
    small_control = (4 + small_control) / 5

    small_velocity = rewards.tolerance(obs[-1], margin=5)
    small_velocity = (1 + small_velocity) / 2
    return (upright * small_control * small_velocity * centered)[0]

_SYSTEM_CONFIG = """
  bodies {
    name: "cart"
    colliders {
      rotation {
      x: 90
      z: 90
      }
      capsule {
        radius: 0.1
        length: 0.4
      }
    }
    frozen { position { x:1 y:1 z:1 } rotation { x:1 y:1 z:1 } }
    mass: 10.
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 1.0
  }
  joints {
    name: "hinge"
    parent: "cart"
    child: "pole"
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    angle_limit { min: -360.0 max: 360.0 }
  }
  actuators {
    name: "hinge"
    joint: "hinge"
    strength: 0.8
    torque {
    }
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.04
  substeps: 8
  dynamics_mode: "pbd"
  """


_SYSTEM_CONFIG_SPRING = """
bodies {
  name: "cart"
  colliders {
    rotation {
    x: 90
    z: 90
    }
    capsule {
      radius: 0.1
      length: 0.2
    }
  }
  frozen { position { x:1 y:1 z:1 } rotation { x:1 y:1 z:1 } }
  mass: 10.
}
bodies {
  name: "pole"
  colliders {
    capsule {
      radius: 0.049
      length: 0.69800085
    }
  }
  frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
  mass: 1.0
}
joints {
  name: "hinge"
  stiffness: 1000.0
  parent: "cart"
  child: "pole"
  child_offset { z: -.3 }
  rotation {
    z: 90.0
  }
  angle_limit {
    min: -360
    max: 360
  }
  limit_strength: 0.0
}
actuators {
  name: "hinge"
  joint: "hinge"
  strength: 0.8
  torque {
  }
}
gravity {
  z: -9.81
}
collide_include {}
dt: 0.04
substeps: 8
dynamics_mode: "legacy_spring"
"""

if __name__=="__main__":

  import jax
  import jax.numpy as jnp
  from brax.io import image
  import matplotlib.pyplot as plt

  key = jax.random.PRNGKey(0)

  pendulum = Pendulum()
  action = jp.ones((pendulum.action_size,)) * 1
  step = jax.jit(pendulum.step)

  rollout = []
  state = pendulum.reset(key)
  for i in range(200):# wiggle sinusoidally
    state = step(state, action)
    rollout.append(state)

  # pendulum.sys.joints[0].angle_vel(qp)

  fig, axes = plt.subplots(3, 1)
  axes[0].plot([pendulum.sys.joints[0].angle_vel(s.qp)[0][0][0] for s in rollout])
  axes[1].plot([jnp.sin(pendulum.sys.joints[0].angle_vel(s.qp)[0][0])[0] for s in rollout])
  axes[2].plot([jnp.cos(pendulum.sys.joints[0].angle_vel(s.qp)[0][0])[0] for s in rollout])
  plt.show()

  im = image.render(pendulum.sys, [s.qp for s in rollout], width=320, height=240)

  import io
  import numpy as np
  from PIL import Image
  from PIL import ImageSequence

  stream_str = io.BytesIO(im)
  images = []
  for frame in ImageSequence.Iterator(Image.open(io.BytesIO(im))):
    images.append(frame)
  images[0].save(
    '/home/palenicek/projects/bb_mbrl/plots/other/pendulum.gif',
    save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0
  )
  print('done')
