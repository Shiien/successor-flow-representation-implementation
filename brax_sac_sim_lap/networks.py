                                  
 
                                                                 
                                                                  
                                         
 
                                                
 
                                                                     
                                                                   
                                                                          
                                                                     
                                

"""SAC networks."""

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
import chex

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
        def __call__(self, feature: jnp.ndarray, task: jnp.ndarray):
                                                                 
                        
                                                                   
            res = []
            for _ in range(self.n_critics):
                q_f = linen.Dense(hidden_layer_sizes[0])(feature)
                q_f = activation(q_f)
                q = linen.Dense(feature_size)(q_f)
                q = activation(q)
                                                                        
                                                          
                q = jnp.einsum("i,...i->...", task, q)[..., None]
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(q_params, feature, task):
        return q_module.apply(q_params, feature, task)

    dummy_obs = jnp.zeros((1, feature_size))
    dummy_task = jnp.zeros((feature_size,))
    return FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_task), apply=apply
    )

def _make_task_network(
    task_size: int,
) -> FeedForwardNetwork:
    class TrainableVector(linen.Module):
        size: int

        @linen.compact
        def __call__(self):
                                                                        
            vector = self.param('vector', linen.initializers.normal(stddev=1.0), (self.size,))
                                                                
            return vector
    
    task_module = TrainableVector(size=task_size)
    
    def apply(task_params):
                                                               
        task_vector = task_module.apply(task_params)
        return task_vector
    
    def init(key):
                                                                      
        return task_module.init(key)
    
    return FeedForwardNetwork(
        init=init, apply=apply
    )

@flax.struct.dataclass
class SACNetworks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution
  feature_network: networks.FeedForwardNetwork
  task_network: networks.FeedForwardNetwork


def make_inference_fn(sac_networks: SACNetworks):
  """Creates params and inference function for the SAC agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

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

def _make_feature_network(
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

    def apply(processor_params, value_params, obs, act, task):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        task = task / (jnp.linalg.norm(task, axis=-1, keepdims=True) + 1e-8)
                                                                 
        chex.assert_equal_rank([obs, act, task])
                                                                       
        obs = jnp.concatenate([obs, act, task], axis=-1)
        feature = value_module.apply(value_params, obs)
        return feature

    obs_size = networks._get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, obs_size + act_size + feature_size))
                                          
    return FeedForwardNetwork(
        init=lambda key: value_module.init(
            key,
            dummy_obs,
        ),
        apply=apply,
    )

make_policy_network = networks.make_policy_network

def make_sac_networks(
    observation_size: int,
    action_size: int,
    feature_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
) -> SACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = make_policy_network(
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
        n_critics=2,
        layer_norm=q_network_layer_norm,
    )
    task_network = _make_task_network(
        feature_size,
    )

    feature_network = _make_feature_network(
        observation_size,
        action_size,
        feature_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
    )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        feature_network=feature_network,
        task_network=task_network,
        parametric_action_distribution=parametric_action_distribution,
    )