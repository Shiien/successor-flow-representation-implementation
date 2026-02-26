import functools
from typing import Any, Callable, Optional, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground import wrapper

class IdTruncationWrapper(wrapper.Wrapper):
  """Automatically resets Brax envs that are done."""

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state = self.env.step(state, action)
    truncation_but_not_done = jp.where(state.done, 1.0 - state.info['truncation'], state.done)
    return state.replace(done = truncation_but_not_done)

def new_wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
):

  env = brax_training.VmapWrapper(env)  

    
  env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
  env = newAutoResetWrapper(env)
  env = IdTruncationWrapper(env)
  return env
  
class newAutoResetWrapper(wrapper.Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = self.env.reset(rng)
    state.info['first_state'] = state.data
    state.info['first_obs'] = state.obs
    state.info["old_obs"] = state.obs
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, state.info['first_state'], state.data)
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    old_state = jax.tree.map(where_done, state.obs, jp.zeros_like(state.obs))
    state = state.replace(data=data, obs=obs)
    state.info.update(old_obs=old_state)
    return state
