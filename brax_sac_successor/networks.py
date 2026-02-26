"""Soft Actor-Critic (SAC) networks with successor features.

This module implements the neural network architectures used in the SAC algorithm
with successor features. It includes networks for:
- Policy network (actor)
- Q-network (critic)
- Psi network (successor features)
- Zeta network (temporal difference learning)

The implementation is based on the paper:
"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
(https://arxiv.org/pdf/1812.05905.pdf)
"""

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.networks import ActivationFn, FeedForwardNetwork, Initializer, MLP
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp

def scalar_positional_embedding(x: jnp.ndarray, d_model: int) -> jnp.ndarray:
    """Create positional embeddings for temporal information.
    
    Args:
        x: Input scalar value
        d_model: Dimension of the embedding
        
    Returns:
        Positional embedding vector
    """
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
    """Creates a Q-network for value estimation.
    
    Args:
        feature_size: Size of input features
        hidden_layer_sizes: Sizes of hidden layers
        activation: Activation function
        n_critics: Number of critics (Q-functions)
        layer_norm: Whether to use layer normalization
        
    Returns:
        FeedForwardNetwork for Q-value estimation
    """
    class QModule(linen.Module):
        """Q-value estimation module."""
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

@flax.struct.dataclass
class SACFlowNetworks:
    """Container for all networks used in SAC with successor features.
    
    Attributes:
        policy_network: Network for policy (actor)
        q_network: Network for Q-values (critic)
        parametric_action_distribution: Distribution for action sampling
        psi_network: Network for successor features
        zeta_network: Network for temporal difference learning
    """
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    psi_network: networks.FeedForwardNetwork
    zeta_network: networks.FeedForwardNetwork


def make_inference_fn(sac_networks: SACFlowNetworks):
    """Creates inference function for the SAC agent.
    
    Args:
        sac_networks: Container of all networks
        
    Returns:
        Function that creates a policy for action selection
    """
    def make_policy(
        params: types.PolicyParams, deterministic: bool = False
    ) -> types.Policy:
        """Creates a policy function.
        
        Args:
            params: Network parameters
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Policy function that maps observations to actions
        """
        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = sac_networks.policy_network.apply(*params, observations)
            if deterministic:
                return sac_networks.parametric_action_distribution.mode(logits), {}
            return (
                sac_networks.parametric_action_distribution.sample(
                    logits, key_sample
                ),
                {},
            )
        return policy
    return make_policy

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
    """Creates a network for successor features (psi).
    
    Args:
        obs_size: Size of observation space
        act_size: Size of action space
        feature_size: Size of feature representation
        preprocess_observations_fn: Function to preprocess observations
        hidden_layer_sizes: Sizes of hidden layers
        activation: Activation function
        layer_norm: Whether to use layer normalization
        obs_key: Key for observation in dictionary
        
    Returns:
        FeedForwardNetwork for successor features
    """
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
        init=lambda key: value_module.init(key, dummy_obs),
        apply=apply,
    )


def _make_zeta_network(
    obs_size: types.ObservationSize,
    emb_size: int,
    feature_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    zeta_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
    obs_key: str = "state",
) -> FeedForwardNetwork:
    """Creates a network for temporal difference learning (zeta).
    
    Args:
        obs_size: Size of observation space
        emb_size: Size of temporal embedding
        feature_size: Size of feature representation
        preprocess_observations_fn: Function to preprocess observations
        hidden_layer_sizes: Sizes of hidden layers
        zeta_hidden_layer_sizes: Sizes of hidden layers for zeta network
        activation: Activation function
        layer_norm: Whether to use layer normalization
        obs_key: Key for observation in dictionary
        
    Returns:
        FeedForwardNetwork for temporal difference learning
    """
    obs_size = networks._get_obs_state_size(obs_size, obs_key)

    class _MLP_t(linen.Module):
        """MLP module with temporal information."""
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

def make_sac_networks(
    observation_size: int,
    action_size: int,
    feature_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    zeta_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
) -> SACFlowNetworks:
    """Creates all networks needed for SAC with successor features.
    
    Args:
        observation_size: Size of observation space
        action_size: Size of action space
        feature_size: Size of feature representation
        preprocess_observations_fn: Function to preprocess observations
        hidden_layer_sizes: Sizes of hidden layers
        zeta_hidden_layer_sizes: Sizes of hidden layers for zeta network
        activation: Activation function
        policy_network_layer_norm: Whether to use layer norm in policy network
        q_network_layer_norm: Whether to use layer norm in Q-network
        
    Returns:
        Container with all networks
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
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
        hidden_layer_sizes=zeta_hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
    )
    return SACFlowNetworks(
        policy_network=policy_network,
        q_network=q_network,
        psi_network=psi_network,
        zeta_network=zeta_network,
        parametric_action_distribution=parametric_action_distribution,
    )