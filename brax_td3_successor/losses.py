from typing import Any, Tuple

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
from . import networks as td3_networks
import rlax

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
    bc: bool = False,
    alpha: float = 2.5,
    gamma_for_su:float = 0.99,
    back_critic_grad: bool = False,
    denoising_steps: int = 1,
):
    """Creates the TD3 losses."""
    policy_network = td3_network.policy_network
    q_network = td3_network.q_network
    psi_network = td3_network.psi_network
    zeta_network = td3_network.zeta_network

    def critic_loss(
        q_params: Params,
        psi_params: Params,
        target_psi_params: Params,
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

        if back_critic_grad:
            feature = psi_network.apply(
                normalizer_params,
                psi_params,
                transitions.observation,
                transitions.action,
            )
        else:
            feature = psi_network.apply(
                normalizer_params,
                target_psi_params,
                transitions.observation,
                transitions.action,
            )

        next_feature = psi_network.apply(
            normalizer_params,
            target_psi_params,
            transitions.next_observation,
            next_actions,
        )
        current_q1_q2 = q_network.apply(q_params, feature)
        next_q1_q2 = q_network.apply(
            target_q_params,
            next_feature,
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
        psi_params: Params,
        target_psi_params: Params,
        normalizer_params: Any,
        transitions: Transition,
    ) -> jnp.ndarray:
        """Calculates the TD3 actor loss."""

        new_actions = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )

        def obs_to_q(
            psi_params, target_psi_params, nromalizer_params, q_params, obs, new_actions
        ):
            feature = psi_network.apply(
                nromalizer_params, psi_params, obs, new_actions
            )
            qs = q_network.apply(q_params, feature)
            return jnp.split(qs, 2, axis=-1)[0][0]

        grad_critic = jax.vmap(
            jax.grad(obs_to_q, argnums=5), in_axes=(None, None, None, None, 0, 0)
        )
        dq_da = grad_critic(
            psi_params,
            target_psi_params,
            normalizer_params,
            q_params,
            transitions.observation,
            new_actions,
        )
        batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0))
        loss = jnp.mean(batch_dpg_learning(new_actions, dq_da))
        return loss

    def psi_zeta_loss(
        psi_params,
        zeta_params,
        q_params,
        target_psi_params,
        target_zeta_params,
        target_q_params,
        target_policy_params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Calculates the TD3 psi loss."""
        key_transition, key_psi = jax.random.split(key)
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

        next_action = policy_network.apply(
            normalizer_params, target_policy_params, transitions.next_observation
        )
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

    return critic_loss, actor_loss, psi_zeta_loss
