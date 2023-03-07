import jax
from model_based_rl.types import *


def build_replay_buffer_functions(
    max_replay_size: int,
    num_envs: int,
    obs_normalizer,
    calculate_empirical_delta_var: bool = False,
    trajectory_window: int = 0,
):

    def build_replay_buffer(transition_init_data, obs_size):
        replay_buffer = ReplayBuffer(
            data=jax.tree_map(lambda x: jnp.zeros(
                (num_envs, max_replay_size // num_envs,) + x.shape[1:]), transition_init_data),
            current_size=jnp.array(0, dtype=jnp.int32),
            current_position=jnp.array(0, dtype=jnp.int32),
            max_size=max_replay_size,
            empirical_delta_var=jnp.zeros(obs_size + 1),
        )
        return replay_buffer

    def update_replay_buffer(
        replay_buffer: ReplayBuffer,
        newdata: jnp.array,
    ) -> ReplayBuffer:

        # Add timestep dimension.
        newdata = jax.tree_map(lambda x: x[:, None], newdata)

        def fill(mem, new): return jax.lax.dynamic_update_slice_in_dim(
            mem, new, replay_buffer.current_position, axis=1)
        new_replay_data = jax.tree_map(fill, replay_buffer.data, newdata)

        new_position = (replay_buffer.current_position +
                        1) % (max_replay_size // num_envs)
        new_size = jnp.minimum(
            replay_buffer.current_size + num_envs, max_replay_size)

        if calculate_empirical_delta_var:
            mask = jnp.expand_dims(jnp.arange(
                max_replay_size) < replay_buffer.current_size, axis=-1)
            delta_states = jnp.concatenate(
                [new_replay_data.o_t - new_replay_data.o_tm1, jnp.expand_dims(new_replay_data.r_t, axis=-1)], axis=-1
            )
            empirical_delta_var = jnp.var(
                delta_states, axis=0, where=mask) + 1e-6
        else:
            empirical_delta_var = replay_buffer.empirical_delta_var

        return ReplayBuffer(
            data=new_replay_data,
            current_position=new_position,
            current_size=new_size,
            max_size=replay_buffer.max_size,
            empirical_delta_var=empirical_delta_var
        )

    def sample_replay_buffer(training_state, replay_buffer, batch_size, num_updates, num_envs):
        key, key_env, key_traj = jax.random.split(training_state.key, 3)
        training_state = training_state.replace(key=key)

        k = trajectory_window

        # Sample the observations from the replay memory
        n_samples = (batch_size * num_updates,)
        curr_size, curr_pos, max_size = replay_buffer.current_size, replay_buffer.current_position, replay_buffer.max_size
        replay_is_full = jnp.float32(curr_size == max_size)

        idx_env = jax.random.randint(
            key_env, n_samples, minval=0, maxval=num_envs)

        # When the replay buffer is full, we first sample shifted indices starting from the current position and then
        # wrap the samples around by modulo.
        idx_traj = jax.random.randint(
            key_traj, n_samples,
            minval=replay_is_full * curr_pos,
            maxval=replay_is_full * (curr_pos+(max_size//num_envs)-k) +
            (1.0-replay_is_full) * ((curr_size//num_envs)-k)
        )
        idx_traj = idx_traj % (max_size // num_envs)

        transitions = jax.tree_map(
            lambda a: a[idx_env, idx_traj], replay_buffer.data)

        # Select the rest of the ongoing trajectories and add to transitions
        if k > 0:
            vselect_timestep = jax.vmap(
                lambda t, a: a[idx_env, (idx_traj + t) %
                               (replay_buffer.max_size // num_envs)],
                in_axes=[0, None],
                out_axes=1
            )
            trajectories = jax.tree_map(lambda a: vselect_timestep(
                jnp.arange(k), a), replay_buffer.data)

            transitions = transitions.replace(
                o_tm1_to_K=trajectories.o_tm1,
                norm_o_tm1_to_K=obs_normalizer.apply(
                    training_state.normalizer_params, trajectories.o_tm1),
                a_tm1_to_K=trajectories.a_tm1,
                o_t_to_K=trajectories.o_t,
                norm_o_t_to_K=obs_normalizer.apply(
                    training_state.normalizer_params, trajectories.o_t),
                log_p_tm1_to_K=trajectories.log_p_tm1,
                r_t_to_K=trajectories.r_t,
                d_t_to_K=trajectories.d_t,
                truncation_t_to_K=trajectories.truncation_t,
            )

        transitions = transitions.replace(
            norm_o_tm1=obs_normalizer.apply(
                training_state.normalizer_params, transitions.o_tm1),
            norm_o_t=obs_normalizer.apply(
                training_state.normalizer_params, transitions.o_t),
        )
        transitions = jax.tree_map(lambda t: jnp.reshape(
            t, [num_updates, -1] + list(t.shape[1:])), transitions)
        return training_state, transitions

    return (
        build_replay_buffer,
        update_replay_buffer,
        sample_replay_buffer,
    )
