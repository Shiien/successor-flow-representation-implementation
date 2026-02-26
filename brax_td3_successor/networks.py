from typing import Sequence, Tuple

import jax.numpy as jnp
from brax.training import networks
from brax.training import types
from brax.training.networks import ActivationFn, FeedForwardNetwork, Initializer, MLP
from brax.training.types import PRNGKey
from flax import linen, struct
import jax


def scalar_positional_embedding(x, d_model):
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    position = x * 10000.0  
    pe = jnp.zeros(d_model)
    pe = pe.at[0::2].set(jnp.sin(position * div_term))
    pe = pe.at[1::2].set(jnp.cos(position * div_term))
    return pe


def _make_rffc_q_network(
    feature_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
) -> FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, feature: jnp.ndarray):
            
            
            res = []
            for _ in range(self.n_critics):
                q_f = linen.Dense(hidden_layer_sizes[0])(feature)
                q_f = activation(q_f)
                q = linen.Dense(1)(q_f)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(q_params, feature):
        return q_module.apply(q_params, feature)

    dummy_obs = jnp.zeros((1, feature_size))
    return FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs), apply=apply
    )


@struct.dataclass
class TD3FlowNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    psi_network: networks.FeedForwardNetwork
    zeta_network: networks.FeedForwardNetwork


def make_inference_fn(td3_networks: TD3FlowNetworks, max_action=1.0):
    """Creates params and inference function for the TD3 agent."""

    def make_policy(
        params: types.PolicyParams,
        exploration_noise,
        noise_clip,
    ) -> types.Policy:
        def policy(
            observations: types.Observation, key_noise: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            actions = td3_networks.policy_network.apply(*params, observations)
            noise = (
                jax.random.normal(key_noise, actions.shape) * exploration_noise
            ).clip(-noise_clip, noise_clip)
            return (actions + noise).clip(-max_action, max_action), {}

        return policy

    return make_policy


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        raw_actions = policy_module.apply(policy_params, obs)
        return linen.tanh(raw_actions)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply
    )


def _make_psi_network(
    obs_size: types.ObservationSize,
    act_size: int,
    feature_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
    obs_key: str = "state",
) -> FeedForwardNetwork:
    """Creates a value network."""
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [feature_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        layer_norm=layer_norm,
    )

    def apply(processor_params, value_params, obs, act):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        obs = jnp.concatenate([obs, act], axis=-1)
        return value_module.apply(value_params, obs)

    obs_size = networks._get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, obs_size + act_size))
    
    return FeedForwardNetwork(
        init=lambda key: value_module.init(
            key,
            dummy_obs,
        ),
        apply=apply,
    )


def _make_zeta_network(
    obs_size: types.ObservationSize,
    emb_size: int,
    feature_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    zeta_hidden_layer_sizes: Sequence[int] = (512, 512),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
    obs_key: str = "state",
) -> FeedForwardNetwork:
    """Creates a value network."""
    obs_size = networks._get_obs_state_size(obs_size, obs_key)

    class _MLP_t(linen.Module):
        """MLP module."""

        layer_sizes: Sequence[int]
        activation: ActivationFn = linen.relu
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
        activate_final: bool = False
        bias: bool = True
        layer_norm: bool = False

        @linen.compact
        def __call__(self, data: jnp.ndarray, t: jnp.ndarray):
            hidden = data
            t = jax.vmap(scalar_positional_embedding, in_axes=(0, None))(t, emb_size)
            hidden = jnp.concatenate([hidden, t], axis=-1)
            for i, hidden_size in enumerate(self.layer_sizes):
                hidden = linen.Dense(
                    hidden_size,
                    name=f"hidden_{i}",
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                if i != len(self.layer_sizes) - 1 or self.activate_final:
                    hidden = self.activation(hidden)
                    if self.layer_norm:
                        hidden = linen.LayerNorm()(hidden)
            return hidden

    def apply(processor_params, value_params, obs, t):
        
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        fs = value_module.apply(value_params, obs, t)
        fs = fs.reshape((*fs.shape[:-1], obs_size, feature_size))
        return fs

    value_module = _MLP_t(
        layer_sizes=list(zeta_hidden_layer_sizes) + [feature_size * obs_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        layer_norm=layer_norm,
    )
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_t = jnp.zeros((1, 1))
    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs, dummy_t), apply=apply
    )


def make_td3_networks(
    observation_size: int,
    action_size: int,
    feature_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    zeta_hidden_layer_sizes: Sequence[int] = (512, 512),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm=False,
    q_network_layer_norm=False,
) -> TD3FlowNetworks:
    """Make TD3 networks."""
    policy_network = make_policy_network(
        action_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
    )

    q_network = _make_rffc_q_network(
        feature_size,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        n_critics=2,
        layer_norm=q_network_layer_norm,
    )

    psi_network = _make_psi_network(
        observation_size,
        action_size,
        feature_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
    )

    zeta_network = _make_zeta_network(
        observation_size,
        64,
        feature_size,
        preprocess_observations_fn=preprocess_observations_fn,
        zeta_hidden_layer_sizes=zeta_hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
    )
    return TD3FlowNetworks(
        policy_network=policy_network,
        q_network=q_network,
        psi_network=psi_network,
        zeta_network=zeta_network,
    )
