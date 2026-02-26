import wandb
import random
import os
from datetime import datetime
import functools
import jax
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from brax_td3 import train as td3
from brax_td3 import networks as td3_networks
from brax_td3_successor import train as td3_successor
from brax_td3_successor import networks as td3_successor_networks
from brax_sac_successor import train as sac_successor   
from brax_sac_successor import networks as sac_successor_networks
from mujoco_playground import registry
from brax_td3_sim import train as td3_sim
from brax_td3_sim import networks as td3_sim_networks
from brax_td3_sim_lap import train as td3_sim_lap
from brax_td3_sim_lap import networks as td3_sim_lap_networks
from brax_sac_sim import train as sac_sim
from brax_sac_sim import networks as sac_sim_networks
from brax_sac_sim_lap import train as sac_sim_lap
from brax_sac_sim_lap import networks as sac_sim_lap_networks

                                          
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
jax.config.update("jax_default_matmul_precision", "highest")

                                    
method_name = "td3"

                               
CUDA_D = {
    "sac": "0",
    "td3": "1",
    "sac_su_2_99": "1",
    "sac_su_2_0": "0",
    "td3_0": "1",
    "td3_99": "0",
    "sac_sim": "0",
    "sac_sim_lap": "1",
    "td3_sim": "0",
    "td3_sim_lap": "1",
}[method_name]

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_D

                         
networks_fn = {
    "td3_0": td3_successor_networks.make_td3_networks,
    "td3_99": td3_successor_networks.make_td3_networks,
    "td3": td3_networks.make_td3_networks,
    "sac": sac_networks.make_sac_networks,
    "sac_su_2_0": sac_successor_networks.make_sac_networks,
    "sac_su_2_99": sac_successor_networks.make_sac_networks,
    "sac_sim": sac_sim_networks.make_sac_networks,
    "sac_sim_lap": sac_sim_lap_networks.make_sac_networks,
    "td3_sim": td3_sim_networks.make_td3_networks,
    "td3_sim_lap": td3_sim_lap_networks.make_td3_networks,
}[method_name]

                           
train_fn_0 = {
    "td3_0": td3_successor.train,
    "td3_99": td3_successor.train,
    "td3": td3.train,
    "sac": sac.train,
    "sac_su_2_0": sac_successor.train,
    "sac_su_2_99": sac_successor.train,
    "sac_sim": sac_sim.train,
    "sac_sim_lap": sac_sim_lap.train,
    "td3_sim": td3_sim.train,
    "td3_sim_lap": td3_sim_lap.train,
}[method_name]

                                     

def train_agent(env_name, seed, feature_size, denoising_steps, tau_zeta):
    """
    Train a reinforcement learning agent with specified parameters.
    
    Args:
        env_name (str): Name of the environment to train on
        seed (int): Random seed for reproducibility
        feature_size (int): Size of the feature representation
        denoising_steps (int): Number of denoising steps
        tau_zeta (float): Tau value for zeta parameter
        
    Returns:
        dict: Training metrics
    """
    env = registry.load(env_name)
    env_config = registry.get_default_config(env_name)
    print("Environment name: ", env_name)
    print("Config: ", env_config)
    print("Action spec: ", env.action_size)
    print("Observation spec: ", env.observation_size)
    
    from mujoco_playground.config import dm_control_suite_params
    print("\n")
    sac_params = dm_control_suite_params.brax_sac_config(env_name)

    x_data, y_data, y_dataerr = [], [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Callback function to track training progress"""
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])
        wandb.log(step=num_steps, data=metrics)

                                   
    sac_training_params = dict(sac_params)
    sac_training_params["num_evals"] = 100
    sac_training_params["normalize_observations"] = True
    
                                             
    if not method_name.startswith("sac"):
        sac_training_params.update(
            policy_delay=1, noise_clip=0.3, smoothing_noise=0.2, exploration_noise=0.2,
        )

    successor_overrides = {
        "td3_99":      {"gamma_for_su": 0.99, "back_critic_grad": True},
        "td3_0":       {"gamma_for_su": 0.0,  "back_critic_grad": True},
        "sac_su_2_99": {"gamma_for_su": 0.99, "use_extra_q_align": True},
        "sac_su_2_0":  {"gamma_for_su": 0.0,  "use_extra_q_align": True},
    }

    if method_name in successor_overrides:
        sac_training_params.update(
            successor_overrides[method_name],
            tau_zeta=tau_zeta,
            denoising_steps=denoising_steps,
        )
    elif method_name not in ("sac", "td3"):
        raise ValueError(f"Method name {method_name} not supported")
            
    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
        
                                    
    make_network_dict = {
        "q_network_layer_norm": True,
        "policy_network_layer_norm": True,
        "hidden_layer_sizes": (512, 512, 512),
    }
    if method_name != "sac" and method_name != "td3":
        make_network_dict["feature_size"] = feature_size
    if method_name in ["sac_su_2_99", "sac_su_2_0", "td3_0", "td3_99"]:
        make_network_dict["zeta_hidden_layer_sizes"] = (512, 512)
    
    if 'Run' in method_name and 'zeta_hidden_layer_sizes' in make_network_dict:
        del make_network_dict['zeta_hidden_layer_sizes']
        make_network_dict["zeta_hidden_layer_sizes"] = (512, 512)
    
    network_factory = functools.partial(
        networks_fn,
        **make_network_dict,
    )

    train_fn = functools.partial(
        train_fn_0,
        **dict(sac_training_params),
        network_factory=network_factory,
        progress_fn=progress,
    )

    from mujoco_playground import wrapper

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        seed=seed,
    )
    
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    return metrics

                     
sweep_config = {
    "method": "grid",  
    "parameters": {
        "z_seed": {
            "values": [0,]
        },
        "feature_size": {
            "values": [512]
        },
        "denoising_steps": {
            "values": [1]
        },
        "tau_zeta": {
            "values": [0.05]
        },
        "env": {
            "values": [
                "AcrobotSwingup",
                "CartpoleBalanceSparse",
                "CartpoleSwingupSparse",
                "CheetahRun",
                "FishSwim",
                "HopperHop",
                "PendulumSwingup",]
        },
    },
}

def train():
    """Main training function that initializes wandb and runs the training process"""
    with wandb.init(name=method_name) as run:
        config = run.config
        env = config.env
        seed = config.z_seed
        feature_size = config.feature_size
        denoising_steps = config.denoising_steps    
        tau_zeta = config.tau_zeta
        train_agent(env, seed, feature_size, denoising_steps, tau_zeta)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="sweep_mujoco_playground_open_source")
    wandb.agent(sweep_id, function=train)
