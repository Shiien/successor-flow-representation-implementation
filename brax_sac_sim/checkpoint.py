import json
from typing import Any, Union

from brax.training import checkpoint
from brax.training import types
from . import networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = "sac_network_config.json"


def save(path: Union[str, epath.Path], step: int, params: Any, config: config_dict.ConfigDict):
    return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(path: Union[str, epath.Path]):
    return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[networks.SACNetworks],
) -> config_dict.ConfigDict:
    return checkpoint.network_config(observation_size, action_size, normalize_observations, network_factory)


def _get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[networks.SACNetworks],
) -> networks.SACNetworks:
    return checkpoint.get_network(config, network_factory)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[networks.SACNetworks] = networks.make_sac_networks,
    deterministic: bool = True,
):
    path = epath.Path(path)
    config_path = path / _CONFIG_FNAME
    if not config_path.exists():
        raise ValueError(f"sac_sim config file not found at {config_path.as_posix()}")

    config = config_dict.create(**json.loads(config_path.read_text()))

    params = load(path)
    nets = _get_network(config, network_factory)
    make_inference_fn = networks.make_inference_fn(nets)
    return make_inference_fn(params, deterministic=deterministic)


