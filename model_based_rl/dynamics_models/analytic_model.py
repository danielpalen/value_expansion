from typing import Optional, Union
import chex

from brax import envs
from brax.training import normalization

from model_based_rl.replay_buffer import *
from model_based_rl.types import *


class DynamicsModel:

    def __init__(
        self,
        env_name: str,
        obs_size: int,
        acts_size: int,
        action_repeat: int,
        normalizer: normalization.Normalizer,
        termination_fn: Callable[[Observation, Action, NextObservation], jnp.ndarray],
        **kwargs,
    ):
        self.name = env_name
        self.acts_size = acts_size
        self.obs_size = obs_size
        self.normalizer = normalizer
        self.action_repeat = action_repeat
        self.termination_fn = termination_fn
        self.grad_loss = jax.value_and_grad(self.loss, has_aux=True)

        self.core_env = envs.create(
            env_name=env_name,
            auto_reset=False,
            # Note: We dont want the episode to terminate. But since the brax episode wrapper handels
            episode_length=jnp.inf,
            # the action_repeat, we need to set this.
            batch_size=1,
            action_repeat=action_repeat,
        )

        assert self.acts_size == self.core_env.action_size
        assert self.obs_size == self.core_env.observation_size
        self._step = jax.jit(self.core_env.step)

    def init(
        self,
        key: PRNGKey
    ) -> Tuple[Params, optax.OptState]:
        return {}, {}

    def loss(
        self,
        dynamics_params: Params,
        normalizer_params: Params,
        transitions: Transition,
    ) -> Tuple[jnp.ndarray, Metrics]:
        return jnp.array(0.0), {}

    def update_step(
        self,
        training_state: TrainingState,
        transitions: Transition
    ) -> Tuple[TrainingState, Metrics]:
        return training_state, {}

    def train(
        self,
        training_state: TrainingState,
        # replay_buffer: ReplayBuffer,
        *args, **kwargs
    ) -> Tuple[TrainingState, Metrics]:
        return training_state, {}

    def apply(
        self,
        params: Tuple[Params, Params],
        obs: Observation, acts: Action
    ) -> jnp.ndarray:
        raise NotImplementedError

    def step(
        self,
        params: Tuple[Params, Params],
        key: PRNGKey,
        state: brax.QP,
        observation: jnp.array,
        norm_observation: Union[Observation, None],
        acts: Action,
    ) -> Transition:
        chex.assert_shape(observation, (None, self.obs_size))
        chex.assert_shape(acts, (None, self.acts_size))
        (dynamics_params, normalizer_params) = params
        num_batch = observation.shape[0]

        if norm_observation is None:
            norm_observation = self.normalizer.apply(
                normalizer_params, observation)

        env_state = brax.envs.State(
            qp=state,
            obs=observation,
            reward=jnp.zeros((num_batch,)),
            done=jnp.zeros((num_batch,)),
            metrics={},
            info={'steps': jnp.zeros((num_batch,)),
                  'truncation': jnp.zeros((num_batch,))}
        )

        next_env_state = self._step(env_state, acts)
        next_state = next_env_state.qp
        next_observation = next_env_state.obs
        next_norm_observation = self.normalizer.apply(
            normalizer_params, next_observation)

        return Transition(
            s_tm1=state,
            o_tm1=observation,
            norm_o_tm1=norm_observation,
            a_tm1=acts,
            log_p_tm1=None,
            s_t=next_state,
            o_t=next_observation,
            norm_o_t=next_norm_observation,
            r_t=next_env_state.reward,
            d_t=jnp.ones_like(next_env_state.done),
            truncation_t=jnp.zeros_like(next_env_state.done),
            o_tm1_to_K=None,
            norm_o_tm1_to_K=None,
            a_tm1_to_K=None,
            o_t_to_K=None,
            norm_o_t_to_K=None,
            log_p_tm1_to_K=None,
            r_t_to_K=None,
            d_t_to_K=None,
            truncation_t_to_K=None,
        )

    def rollout(
        self,
        key: PRNGKey,
        state: brax.QP,
        observation: Observation,
        norm_observation: Union[Observation, None],
        policy_params: Tuple[Params, Params],
        dynamics_params: Tuple[Params, Params],
        policy: Callable[[Tuple[Params, Params], PRNGKey, Observation], Action],
        num_steps: Optional[int] = 1,
    ) -> Tuple[Transition, jnp.ndarray]:
        chex.assert_rank(observation, 2)

        if norm_observation is None:
            _, normalizer_params = dynamics_params
            norm_observation = self.normalizer.apply(
                normalizer_params, observation)

        def step(input, key):
            state, obs, norm_obs = input
            key_policy, key_dynamics = jax.random.split(key)
            action, log_prob_action = policy(policy_params, key_policy, obs)
            transition = self.step(
                dynamics_params, key_dynamics, state, obs, norm_obs, action)
            transition = transition.replace(log_p_tm1=log_prob_action)
            return (transition.s_t, transition.o_t, transition.norm_o_t), (transition, log_prob_action)

        # The carry needs to be chose depending on if the step function returns the simulator state or not (brax vs. network model).
        init = (state, observation, norm_observation)
        if step(init, key)[0][0] is None:
            init = (None, observation, norm_observation)

        # Transition with elements: traj.o_t.shape = (k_steps, n_starting_states, -1)
        _, (trajectories, log_prob_action) = jax.lax.scan(
            step, init, jax.random.split(key, num_steps))

        # Update the (not) done flag with the TRUE termination function:
        done = self.termination_fn(
            trajectories.o_tm1, trajectories.a_tm1, trajectories.o_t)
        trajectories = trajectories.replace(d_t=1 - done)
        return trajectories, log_prob_action

    def rollout_transitions(
        self,
        key: PRNGKey,

        transitions: Transition,

        policy_params: Tuple[Params, Params],
        dynamics_params: Tuple[Params, Params],
        policy: Callable[[Tuple[Params, Params], PRNGKey, Observation], Action],
        num_steps: Optional[int] = 1,
        start_at_t: bool = True,
    ) -> Tuple[Transition, jnp.ndarray]:

        batch_size = transitions.o_t.shape[0]
        rollout_key, policy_H_key = jax.random.split(key)

        if start_at_t:
            model_rollout, log_probs = self.rollout(
                key=rollout_key,
                state=transitions.s_t,
                observation=transitions.o_t,
                norm_observation=transitions.norm_o_t,
                policy_params=policy_params,
                dynamics_params=dynamics_params,
                policy=policy,
                num_steps=num_steps,
            )

            # analytical model
            if model_rollout.s_t is None:
                transitions = transitions.replace(s_t=None, s_tm1=None)

            # (s_0, a_0, r_0, s_1) + [(s_t, a_t, r_t, s_{t+1})]_t=1^H
            rollout = jax.tree_multimap(lambda a, b: jnp.concatenate(
                [a[None], b], axis=0), transitions, model_rollout)
            log_probs = jnp.concatenate(
                [jnp.zeros((1, batch_size)), log_probs], axis=0)

        else:
            if num_steps == 0:
                raise AttributeError(
                    "This does not work for num_steps=0, because of the indexing that follows.")

            rollout, log_probs = self.rollout(
                key=rollout_key,
                state=transitions.s_tm1,
                observation=transitions.o_tm1,
                norm_observation=transitions.norm_o_tm1,
                policy_params=policy_params,
                dynamics_params=dynamics_params,
                policy=policy,
                num_steps=num_steps,
            )

        # Append rollout by zeroes dummy timestep and then overwrite relevant positions.
        rollout = jax.tree_map(lambda x: jnp.concatenate(
            [x, jnp.zeros((1, *x.shape[1:]))], axis=0), rollout)

        # Calculate action for last observation
        # -2, because last one is dummy now.
        a_H, log_prob_H = policy(policy_params, policy_H_key, rollout.o_t[-2])
        rollout = rollout.replace(
            o_tm1=jnp.concatenate(
                [rollout.o_tm1[:-1], rollout.o_t[-2:-1]], axis=0),
            norm_o_tm1=jnp.concatenate(
                [rollout.norm_o_tm1[:-1], rollout.norm_o_t[-2:-1]], axis=0),
            a_tm1=jnp.concatenate([rollout.a_tm1[:-1], a_H[None]], axis=0),
            log_p_tm1=jnp.concatenate([log_probs, log_prob_H[None]], axis=0),
        )

        # swap (time, batch, ...) to (batch, time, ...)
        rollout = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), rollout)
        transitions = transitions.replace(
            norm_o_tm1_to_K=rollout.norm_o_tm1,
            o_tm1_to_K=rollout.o_tm1,
            a_tm1_to_K=rollout.a_tm1,
            norm_o_t_to_K=rollout.norm_o_t,
            o_t_to_K=rollout.o_t,
            log_p_tm1_to_K=rollout.log_p_tm1,
            r_t_to_K=rollout.r_t,
            d_t_to_K=rollout.d_t,
            truncation_t_to_K=rollout.truncation_t,
        )

        return transitions
