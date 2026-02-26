













"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any

from brax.training import types
from . import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp
import einops

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
    sac_network: sac_networks.SACFlowNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    gamma_for_su: float=0.99,
    use_extra_q_align: bool=False,
    denoising_steps: int = 1,
):
  """Creates the SAC losses."""

  target_entropy = -0.5 * action_size
  policy_network = sac_network.policy_network
  q_network = sac_network.q_network
  psi_network = sac_network.psi_network
  zeta_network = sac_network.zeta_network
  parametric_action_distribution = sac_network.parametric_action_distribution

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
      psi_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      target_psi_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    
    q_old_feature = psi_network.apply(
        normalizer_params, psi_params, transitions.observation, transitions.action
    )
    q_old_action = q_network.apply(
        q_params, q_old_feature
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
    next_q_feature = psi_network.apply(
        normalizer_params, target_psi_params, transitions.next_observation, next_action)
    next_q = q_network.apply(
        target_q_params,
        next_q_feature
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
      psi_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key
    )
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)
    q_feature = psi_network.apply(
        normalizer_params, psi_params, transitions.observation, action)
    q_action = q_network.apply(
        q_params, q_feature
    )
    min_q = jnp.min(q_action, axis=-1)
    actor_loss = alpha * log_prob - min_q
    return jnp.mean(actor_loss)

  def psi_zeta_loss(
    psi_params,
    zeta_params,
    q_params,
    target_psi_params,
    target_zeta_params,
    target_q_params,
    policy_params,
    alpha,
    normalizer_params: Any,
    transitions: Transition,
    key: PRNGKey,
  ) -> jnp.ndarray:
    """Calculates the TD3 psi loss."""
    key_transition, key_psi, key_action, key_next = jax.random.split(key, 4)
    eps = jax.random.normal(key_transition, transitions.observation.shape)
    t_ = jax.random.uniform(
        key_psi, shape=(transitions.observation.shape[0],), minval=0.0, maxval=1.0
    )
    next_obs = transitions.next_observation
    next_obs = normalize(next_obs, normalizer_params)
    obs_t = t_[:, None] * next_obs + eps * (1.0 - t_[:, None])
    obs_target = next_obs - eps

    z_psi = psi_network.apply(
        normalizer_params, psi_params, transitions.observation, transitions.action
    )
    z_zeta = zeta_network.apply(normalizer_params, zeta_params, obs_t, t_[:, None])
    score = einops.einsum(
        z_psi,
        z_zeta,
        "b c, b d c -> b d",
    )
    score_loss_next = (1.0-gamma_for_su)*jnp.sum(jnp.square(score-obs_target),axis=-1)
    
    def gen_state_diff_with_x(z_psi_, x, t_):
        T = denoising_steps
        time_steps = jnp.linspace(0, 1, T + 1)
        model_fn = lambda x, t: einops.einsum(
            z_psi_,
            zeta_network.apply(normalizer_params, target_zeta_params, x, t),
            "b c, b d c -> b d",
        )
        def _one_step(x, t_star, t_end):
            middle = 0.5 * (t_star + t_end)
            dx = model_fn(x + model_fn(x, t_star) * (t_end - t_star) / 2, middle)
            x = x + dx * (t_end - t_star)
            
            return x

        for t in range(T + 1):
            x = _one_step(
                x, time_steps[t] * t_[:, None], time_steps[t + 1] * t_[:, None]
            )
        return x

    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key_action
    )
    next_action = parametric_action_distribution.postprocess(next_action)
    psi_target_next = psi_network.apply(
        normalizer_params, target_psi_params, transitions.next_observation, next_action
    )

    gen_x = gen_state_diff_with_x(psi_target_next, eps, t_)
    gen_x = gen_x
    zeta_gen_x_target = jax.lax.stop_gradient(
        zeta_network.apply(
            normalizer_params, target_zeta_params, gen_x, t_[:, None]
        )
    )
    zeta_gen_x = zeta_network.apply(
        normalizer_params, zeta_params, gen_x, t_[:, None]
    )
    score_gen = einops.einsum(
        psi_target_next,
        zeta_gen_x_target,
        "b c, b d c -> b d",
    )
    score_new = einops.einsum(
        z_psi,
        zeta_gen_x,
        "b c, b d c -> b d",
    )
    
    score_loss_gen = (
        gamma_for_su
        * jnp.sum(jnp.square(score_new - jax.lax.stop_gradient(score_gen)),axis=-1)
    )
    score_loss = score_loss_next + score_loss_gen
    truncation = transitions.extras['state_extras']['truncation']
    score_loss *= jnp.expand_dims(1 - truncation, -1)
    score_loss = jnp.mean(score_loss)
    
    
    
    
    
    
    
    
    
    
    
    
    
    return score_loss
  return alpha_loss, critic_loss, actor_loss, psi_zeta_loss