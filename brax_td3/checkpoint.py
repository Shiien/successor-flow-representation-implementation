













"""Checkpointing for TD3."""

import json
from typing import Any, Union

from brax.training import checkpoint
from brax.training import types
from . import networks as td3_networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = "td3_network_config.json"


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
    """Saves a checkpoint."""
    return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
    """Loads td3 checkpoint."""
    return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[td3_networks.TD3Networks],
) -> config_dict.ConfigDict:
    """Returns a config dict for re-creating a network from a checkpoint."""
    return checkpoint.network_config(
        observation_size, action_size, normalize_observations, network_factory
    )


def _get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[td3_networks.TD3Networks],
) -> td3_networks.TD3Networks:
    """Generates a td3 network given config."""
    return checkpoint.get_network(
        config, network_factory
    )  


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        td3_networks.TD3Networks
    ] = td3_networks.make_td3_networks,
    deterministic: bool = True,
):
    """Loads policy inference function from td3 checkpoint."""
    path = epath.Path(path)

    config_path = path / _CONFIG_FNAME
    if not config_path.exists():
        raise ValueError(f"td3 config file not found at {config_path.as_posix()}")

    config = config_dict.create(**json.loads(config_path.read_text()))

    params = load(path)
    td3_network = _get_network(config, network_factory)
    make_inference_fn = td3_networks.make_inference_fn(td3_network)

    return make_inference_fn(params, deterministic=deterministic)
