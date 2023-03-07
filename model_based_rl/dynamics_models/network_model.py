from typing import Union, NamedTuple, Optional
import haiku as hk
import optax
import chex
import jax

from brax.training import networks
from brax.training import normalization

from model_based_rl.dynamics_models import analytic_model
from model_based_rl.replay_buffer import *
from model_based_rl.types import *
from model_based_rl.utils import nonlinearity


class DynamicsModel(analytic_model.DynamicsModel):

    def __init__(
        self,
        ensemble_size: int,
        obs_size: int,
        acts_size: int,
        learning_rate: float,
        batch_size: int,
        n_epochs: float,
        deterministic: bool,
        logvar_learned: bool,
        num_updates: int,
        min_updates: int,
        threshold: float,
        termination_fn: Callable[[Observation, Action, NextObservation], jnp.ndarray],
        sample_replay_buffer_fn: Callable,
        normalizer: normalization.Normalizer,
        logvar_limits: Optional[Tuple[float, float]] = (-10., 0.5),
        hidden_layer_sizes: Optional[Tuple[int, ...]] = (256, 256),
        weight_distribution: Optional[str] = 'truncated_normal',
        activation: Optional[str] = 'relu',
        **kwargs,
    ):
        self.ensemble_size = ensemble_size
        self.obs_size = obs_size
        self.acts_size = acts_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.deterministic = deterministic
        self.empirical_inv_var_weighting = deterministic
        self.logvar_learned = logvar_learned and (not deterministic)
        self.logvar_limits = logvar_limits if not deterministic else (0.0, 0.0)
        self.num_updates = num_updates
        self.min_updates = min_updates
        self.threshold = threshold
        self.hidden_layer_sizes = hidden_layer_sizes
        self.weight_distribution = weight_distribution
        self.activation = activation

        self.termination_fn = termination_fn
        self.grad_loss = jax.value_and_grad(self.loss, has_aux=True)
        self.normalizer = normalizer
        self.ensemble = self._build_ensemble()
        self.optimizer = optax.adam(learning_rate=learning_rate)

        self.sample_replay_buffer = sample_replay_buffer_fn

    def init(
        self,
        key: PRNGKey
    ) -> Tuple[Params, optax.OptState]:
        ensemble_params = self.ensemble.init(key)
        optimizer_state = self.optimizer.init(ensemble_params)
        return ensemble_params, optimizer_state

    def _build_ensemble(
        self,
    ) -> networks.FeedForwardModel:
        # INPUT:  [obs + acts]  # normalized
        # OUTPUT: [delta_next_obs + rew] # unnormalized

        dummy_action, dummy_obs = jnp.zeros(
            (1, self.acts_size)), jnp.zeros((1, self.obs_size))
        output_size = (self.obs_size + 1) * 2

        # def dynamics_model_module(obs: jnp.ndarray, actions: jnp.ndarray):
        def dynamics_model_module(obs: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            net = hk.nets.MLP(
                name="dynamics_model",
                activation=nonlinearity[self.activation],
                output_sizes=self.hidden_layer_sizes + (output_size,),
                w_init=hk.initializers.VarianceScaling(
                    scale=1.0, mode='fan_in', distribution=self.weight_distribution),
            )
            pred = net(jnp.concatenate([obs, actions], axis=-1))
            mean, logvar = pred.split(2, axis=-1)
            return mean, logvar

        dynamics_model = hk.without_apply_rng(
            hk.transform(dynamics_model_module))

        # Create Ensemble using single model
        ensemble = networks.FeedForwardModel(
            init=lambda key: jax.vmap(dynamics_model.init, in_axes=[0, None, None])(
                jax.random.split(key, self.ensemble_size), dummy_obs, dummy_action),
            apply=jax.vmap(dynamics_model.apply, in_axes=[
                0, None, None], out_axes=0)
        )

        # Bound logvar prediction
        def logvar_transform_init(key):
            params = {'ensemble': ensemble.init(key)}
            if self.logvar_learned:
                min_logvar = jnp.ones(
                    (1, 1, output_size // 2)) * self.logvar_limits[0]
                max_logvar = jnp.ones(
                    (1, 1, output_size // 2)) * self.logvar_limits[1]
                params['log_transform'] = {
                    'min': min_logvar, 'max': max_logvar}
            return params

        def logvar_transform(params: Params, obs: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            mean, logvar = ensemble.apply(params['ensemble'], obs, actions)

            if self.deterministic:
                logvar = jnp.zeros_like(logvar)
            else:
                # Formular from PETS Appendix A.1 (https://arxiv.org/pdf/1805.12114.pdf)
                min_logvar, max_logvar = self._get_logvar_limits(params)
                logvar = max_logvar - jax.nn.softplus(max_logvar - logvar)
                logvar = min_logvar + jax.nn.softplus(logvar - min_logvar)
            return mean, logvar

        return networks.FeedForwardModel(init=logvar_transform_init, apply=logvar_transform)

    def _get_logvar_limits(self, dynamics_params: Params) -> Tuple[jnp.array, jnp.array]:
        if self.logvar_learned:
            return dynamics_params['log_transform']['min'], dynamics_params['log_transform']['max']
        else:
            return self.logvar_limits

    def loss(
        self,
        dynamics_params: Params,
        normalizer_params: Params,
        transitions: Transition,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[jnp.ndarray, Metrics]:

        observation = transitions.o_tm1
        next_observation = transitions.o_t
        action = transitions.a_tm1
        delta_states = next_observation - observation
        rewards = jnp.expand_dims(transitions.r_t, -1)

        not_done = jnp.expand_dims(transitions.d_t, axis=0)
        not_done = jnp.ones_like(not_done)

        targets = jnp.concatenate([delta_states, rewards], axis=-1)
        num_not_done = jnp.sum(not_done)

        # Check input shapes:
        batch_size, n_obs = observation.shape
        chex.assert_shape(targets, (batch_size, self.obs_size + 1))
        chex.assert_shape(observation, (batch_size, self.obs_size))
        chex.assert_shape(next_observation, (batch_size, self.obs_size))
        chex.assert_shape(action, (batch_size, self.acts_size))

        normalized_observation = self.normalizer.apply(
            normalizer_params, observation)
        mean, logvar = self.ensemble.apply(
            dynamics_params, normalized_observation, action)

        if self.empirical_inv_var_weighting:
            inv_var = 1. / \
                jnp.expand_dims(replay_buffer.empirical_delta_var, axis=(0, 1))
        else:
            inv_var = jnp.exp(-logvar)

        squared_error = jnp.square(mean-targets)
        mse_loss = jnp.sum(squared_error * inv_var, axis=-1)
        entropy = jnp.sum(logvar, axis=-1)

        loss = negative_log_likelihood = jnp.mean(entropy + mse_loss)

        min_logvar, max_logvar = self._get_logvar_limits(dynamics_params)

        if self.logvar_learned:
            loss = loss + 0.01 * (jnp.sum(max_logvar) - jnp.sum(min_logvar))

        chex.assert_shape(not_done, (1, batch_size))
        chex.assert_shape(
            mean, (self.ensemble_size, batch_size, self.obs_size + 1))
        chex.assert_shape(logvar, (self.ensemble_size,
                          batch_size, self.obs_size + 1))
        chex.assert_shape([mse_loss, entropy],
                          (self.ensemble_size, batch_size))

        metrics = {
            'dynamics_model/loss': loss,
            'dynamics_model/neg_loglikelihood': negative_log_likelihood,
            'dynamics_model/mse': jnp.mean(squared_error),
            'dynamics_model/weighted_mse': jnp.mean(mse_loss),
            'dynamics_model/entropy': jnp.mean(entropy),
            'dynamics_model/num_not_done': num_not_done / batch_size,
            'dynamics_model/max_logvar': jnp.mean(max_logvar),
            'dynamics_model/min_logvar': jnp.mean(min_logvar),
            'dynamics_model/logvar': jnp.mean(logvar),
            'dynamics_model/inv_var': jnp.mean(inv_var),
        }

        return loss, metrics

    def update_step(
        self,
        training_state: TrainingState,
        transitions: Transition,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[TrainingState, Metrics]:
        (_, metrics), grad = self.grad_loss(training_state.dynamics_model_params, training_state.normalizer_params,
                                            transitions, replay_buffer)

        params_update, new_optim_state = self.optimizer.update(
            grad, training_state.dynamics_optimizer_state)
        new_params = optax.apply_updates(
            training_state.dynamics_model_params, params_update)

        new_training_state = training_state.replace(
            dynamics_optimizer_state=new_optim_state,
            dynamics_model_params=new_params,
        )
        return new_training_state, metrics

    def train(
        self,
        training_state: TrainingState,
        replay_buffer: ReplayBuffer,
        n_epochs: int = None,
        *args, **kwargs
    ) -> Tuple[TrainingState, Metrics]:

        if n_epochs is None:
            n_epochs = self.n_epochs

        training_state, eval_transitions = self.sample_replay_buffer(
            training_state, replay_buffer, int(1048576 * 0.1), 1)
        eval_transitions = jax.tree_map(lambda x: x[0], eval_transitions)

        # Training
        def epoch_step(_, training_state):
            training_state, transitions_i = self.sample_replay_buffer(
                training_state, replay_buffer, self.batch_size, 1)
            training_state, next_metrics = self.update_step(
                training_state, jax.tree_map(lambda x: x[0], transitions_i), replay_buffer)
            return training_state

        n_mini_batches = jnp.int32(
            replay_buffer.current_size / self.batch_size * n_epochs)
        training_state = jax.lax.fori_loop(
            0, n_mini_batches, epoch_step, training_state)

        # Evaluation
        _, eval_after = self.loss(training_state.dynamics_model_params,
                                  training_state.normalizer_params, eval_transitions, replay_buffer)

        metrics = {
            'dynamics_model/n_grad_steps': n_mini_batches,
            **{f"{k}_eval": v for k, v in eval_after.items()}
        }

        return training_state, metrics

    def step(
        self,
        params: Tuple[Params, Params],
        key: PRNGKey,
        state: Union[brax.QP, None],
        obs: Union[Observation, None],
        norm_obs: Union[Observation, None],
        acts: Action,
    ) -> Transition:

        # Assert that either obs or norm_obs is present.
        # It is optimal to pass both obs & norm_obs.
        assert obs is not None or norm_obs is not None

        ensemble_key, gaussian_key = jax.random.split(key)
        (dynamics_params, normalizer_params) = params

        if norm_obs is None:
            norm_obs = self.normalizer.apply(normalizer_params, obs)

        if obs is None:
            obs = self.normalizer.inverse(normalizer_params, norm_obs)

        mean, logvar = self.ensemble.apply(dynamics_params, norm_obs, acts)

        # Generate Gaussian samples
        if not self.deterministic:
            std = jnp.sqrt(jnp.exp(logvar))
            pred = mean + std * jax.random.normal(gaussian_key, mean.shape)

        else:
            pred = mean

        # Aggregate ensemble dimension by uniform sampling
        ensemble_size, batch_size, _ = pred.shape
        model_inds = jax.random.choice(
            ensemble_key, jnp.arange(ensemble_size), (batch_size,))
        batch_inds = jnp.arange(batch_size)
        pred = pred[model_inds, batch_inds]
        obs_delta, rewards = pred[:, :-1], pred[:, -1]

        # Predict the UNNORMALIZED delta:
        next_obs = obs + obs_delta
        next_norm_obs = self.normalizer.apply(normalizer_params, next_obs)

        return Transition(
            s_tm1=None,
            o_tm1=obs,
            norm_o_tm1=norm_obs,
            a_tm1=acts,
            log_p_tm1=None,
            s_t=None,
            o_t=next_obs,
            norm_o_t=next_norm_obs,
            r_t=rewards,
            d_t=jnp.ones_like(rewards),
            truncation_t=jnp.zeros_like(rewards),
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
