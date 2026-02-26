from typing import Any, Tuple

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
from . import networks as td3_networks
import rlax
import chex
Transition = types.Transition

import einops

def normalize(batch,
              mean_std,
              max_abs_value=None):
  """Normalizes data using running statistics."""

  def normalize_leaf(data: jnp.ndarray, mean: jnp.ndarray,
                     std: jnp.ndarray) -> jnp.ndarray:
                            
    if not jnp.issubdtype(data.dtype, jnp.inexact):
      return data
    data = (data - mean) / std
    if max_abs_value is not None:
                                     
      data = jnp.clip(data, -max_abs_value, +max_abs_value)
    return data

  return jax.tree_util.tree_map(normalize_leaf, batch, mean_std.mean, mean_std.std)
  
def make_losses(
    td3_network: td3_networks.TD3FlowNetworks,
    reward_scaling: float,
    discounting: float,
    smoothing: float,
    noise_clip: float,
    max_action: float = 1.0,
):
    """Creates the TD3 losses."""
    policy_network = td3_network.policy_network
    q_network = td3_network.q_network
    feature_network = td3_network.feature_network
    task_network = td3_network.task_network

    def critic_loss(
        q_params: Params,
        feature_params: Params,
        task_params: Params,
        target_q_params: Params,
        target_policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Calculates the TD3 critic loss."""

        next_actions = policy_network.apply(
            normalizer_params, target_policy_params, transitions.next_observation
        )
        smoothing_noise = (jax.random.normal(key, next_actions.shape) * smoothing).clip(
            -noise_clip, noise_clip
        )
        next_actions = (next_actions + smoothing_noise).clip(-max_action, max_action)

        feature = feature_network.apply(
            normalizer_params,
            feature_params,
            transitions.observation,
            transitions.action,
            transitions.extras['task']
        )

        next_feature = feature_network.apply(
            normalizer_params,
            feature_params,
            transitions.next_observation,
            next_actions,
            transitions.extras['task']
        )
        task = task_network.apply(task_params)
        current_q1_q2 = q_network.apply(q_params, feature, task)
        next_q1_q2 = q_network.apply(
            target_q_params,
            next_feature,
            task,
        )
        target_q = jnp.min(next_q1_q2, axis=-1)
        
        
        
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling
            + transitions.discount * discounting * target_q
        )

        q_error = current_q1_q2 - jnp.expand_dims(target_q, -1)

                                                     
        truncation = transitions.extras['state_extras']['truncation']
        q_error *= jnp.expand_dims(1 - truncation, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss

    def actor_loss(
        policy_params: Params,
        q_params: Params,
        feature_params: Params,
        task_params: Params,
        normalizer_params: Any,
        transitions: Transition,
    ) -> jnp.ndarray:
        """Calculates the TD3 actor loss."""

        new_actions = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        def obs_to_q(
            feature_params, task_params, nromalizer_params, q_params, obs, new_actions, feature_task
        ):
            task = task_network.apply(task_params)
            feature = feature_network.apply(
                nromalizer_params, feature_params, obs, new_actions, feature_task
            )
            qs = q_network.apply(q_params, feature, task)
            return jnp.split(qs, 2, axis=-1)[0][0]

        grad_critic = jax.vmap(
            jax.grad(obs_to_q, argnums=5), in_axes=(None, None, None, None, 0, 0, 0)
        )
        dq_da = grad_critic(
            feature_params,
            task_params,
            normalizer_params,
            q_params,
            transitions.observation,
            new_actions,
            transitions.extras['task']
        )
        batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0))
        loss = jnp.mean(batch_dpg_learning(new_actions, dq_da))
        return loss

    def task_loss(
        task_params,
        feature_params,
        target_policy_params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Calculates the TD3 psi loss."""
        key_transition, key_psi = jax.random.split(key)
        next_actions = policy_network.apply(
            normalizer_params, target_policy_params, transitions.next_observation
        )
        smoothing_noise = (jax.random.normal(key, next_actions.shape) * smoothing).clip(
            -noise_clip, noise_clip
        )
        next_actions = (next_actions + smoothing_noise).clip(-max_action, max_action)
                  
        next_feature = feature_network.apply(
            normalizer_params,
            feature_params,
            transitions.next_observation,
            next_actions,
            transitions.extras['task']
        )
        task = task_network.apply(task_params)
        reward_prediction = jnp.einsum("i,...i->...", task, next_feature)
        truncation = transitions.extras['state_extras']['truncation']
        chex.assert_equal_shape([reward_prediction, transitions.reward])
        reward_prediction_loss = jnp.mean(jnp.square(reward_prediction - transitions.reward)*(1 - truncation))
        return reward_prediction_loss

    return critic_loss, actor_loss, task_loss
