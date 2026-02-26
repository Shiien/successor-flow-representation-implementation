from typing import Any, Tuple

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey
from . import networks
import chex
Transition = types.Transition


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
    sac_nets: networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
    target_entropy = -0.5 * action_size
    policy_network = sac_nets.policy_network
    q_network = sac_nets.q_network
    feature_network = sac_nets.feature_network
    task_network = sac_nets.task_network
    parametric_action_distribution = sac_nets.parametric_action_distribution
    def alpha_loss(
      log_alpha: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)

    def critic_loss(
        q_params: Params,
        feature_params: Params,
        policy_params: Params,
        task_params: Params,
        target_q_params: Params,
        target_feature_params: Params,
        normalizer_params: Any,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:

        task = task_network.apply(task_params)
        q_old_feature = feature_network.apply(
            normalizer_params, 
            feature_params, 
            transitions.observation, 
            transitions.action, 
            transitions.extras['task']
        )
        q_old_action = q_network.apply(
            q_params, q_old_feature, task
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q_feature = feature_network.apply(
            normalizer_params, target_feature_params, transitions.next_observation, next_action, transitions.extras['task']
        )
        next_q = q_network.apply(
            target_q_params,
            next_q_feature,
            task
        )
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling
            + transitions.discount * discounting * next_v
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

                                                      
        truncation = transitions.extras['state_extras']['truncation']
        q_error *= jnp.expand_dims(1 - truncation, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        feature_params: Params,
        task_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        task = task_network.apply(task_params)
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_feature = feature_network.apply(
            normalizer_params, feature_params, transitions.observation, action, transitions.extras['task']
        )
        q_action = q_network.apply(
            q_params, q_feature, task
        )
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q
        return jnp.mean(actor_loss)

    def task_loss(
        task_params,
        feature_params,
        policy_params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        key_transition, key_psi = jax.random.split(key)
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        
        next_feature = feature_network.apply(
            normalizer_params,
            feature_params,
            transitions.next_observation,
            next_action,
            transitions.extras['task']
        )
        task = task_network.apply(task_params)
        reward_prediction = jnp.einsum("i,...i->...", task, next_feature)
        truncation = transitions.extras['state_extras']['truncation']
        chex.assert_equal_shape([reward_prediction, transitions.reward])
        reward_prediction_loss = jnp.mean(jnp.square(reward_prediction - transitions.reward)*(1 - truncation))
        return reward_prediction_loss
    
    def lap_loss(
        feature_params,
        normalizer_params,
        policy_params,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        current_feature = feature_network.apply(
            normalizer_params,
            feature_params,
            transitions.observation,
            transitions.action,
            transitions.extras['task']
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_feature = feature_network.apply(
            normalizer_params,
            feature_params,
            transitions.next_observation,
            next_action,
            transitions.extras['task']
        )
        current_feature_norm = current_feature / (jnp.linalg.norm(current_feature, axis=-1, keepdims=True) + 1e-8)
        next_feature_norm = next_feature / (jnp.linalg.norm(next_feature, axis=-1, keepdims=True) + 1e-8)
        laplacian_loss = jnp.mean(jnp.square(current_feature_norm - next_feature_norm))
        Cov = jnp.matmul(current_feature_norm, next_feature_norm.T)
        I = jnp.eye(Cov.shape[0], dtype=jnp.bool_)
        chex.assert_equal_shape([Cov, I])
        off_diag = ~I
        orth_loss_diag = -2 * jnp.diag(Cov).mean()
        off_diag_mask = off_diag.astype(jnp.float32)
        orth_loss_offdiag = (Cov**2 * off_diag_mask).sum() / off_diag_mask.sum()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        return (laplacian_loss + orth_loss)

    return critic_loss, actor_loss, alpha_loss, task_loss, lap_loss


