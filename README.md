# Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

This repository contains implementations of reinforcement learning algorithms with successor flow features, specifically focusing on Soft Actor-Critic (SAC) and TD3 variants. The code is built on top of the Brax physics engine and JAX for efficient computation.

## Overview

The project implements several reinforcement learning algorithms:
- SAC (Soft Actor-Critic)
- SAC with Successor Flow Features
- SAC with simple successor feature
- SAC with simple successor feature with Laplacian
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)
- TD3 with Successor Flow Features
- TD3 with simple successor feature
- TD3 with simple successor feature with Laplacian


The successor features implementation enables better transfer learning and representation learning capabilities by learning temporal abstractions of the environment dynamics.

## Installation

Create and activate a conda environment, then install dependencies:

```bash
conda create -n sf2 python=3.11 -y
conda activate sf2
pip install --upgrade pip
pip install -r requirements.txt
```

### JAX with GPU (CUDA 12)

This project installs JAX with NVIDIA GPU support via the **official pip extra** (CUDA and cuDNN from pip wheels). See [JAX installation — NVIDIA GPU](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu).

- **NVIDIA driver:** On Linux, use driver version **≥ 525** for CUDA 12.
- **Do not set `LD_LIBRARY_PATH`** when using these wheels; it can override the bundled CUDA libraries and cause *Error loading CUDA libraries* or *DNN library initialization failed*. Unset it if needed: `unset LD_LIBRARY_PATH`.
- **CPU-only:** In `requirements.txt`, replace the line `jax[cuda12]==0.6.0` with `jax==0.6.0` and `jaxlib==0.6.0`, then reinstall.

## Project Structure

```
submission/
├── requirements.txt          # Project dependencies
├── sweep.py                 # Hyperparameter sweep configuration
├── brax_sac_successor/      # SAC with successor flow features implementation
│   ├── __init__.py
│   ├── networks.py          # Neural network architectures
│   ├── checkpoints.py       # Model checkpointing utilities
│   ├── losses.py           # Loss functions
│   └── train.py            # Training loop implementation
├── brax_td3/               # TD3 implementation
├── brax_td3_successor/     # TD3 with successor flow features
|   ...
└── README.md               # This file
```

## Usage

### Training

To run a hyperparameter sweep:

```bash
python sweep.py
```

The sweep configuration in `sweep.py` can be modified to experiment with different:
- Feature sizes
- Denoising steps
- Tau values
- Environments
- Random seeds

### Key Parameters

- `feature_size`: Size of the feature representation
- `denoising_steps`: Number of denoising steps for successor flow features
- `tau_zeta`: Update rate for zeta network
- `gamma_for_su`: Discount factor for successor flow features

## Dependencies

Key dependencies include:
- JAX and JAXlib for numerical computing
- Flax for neural networks
- Brax and MujocoPlayground for physics simulation
- Optax for optimization
- Wandb for experiment tracking

See `requirements.txt` for a complete list of dependencies.



## Citation
If you use this code for your research, please cite our paper
```
@inproceedings{
    shi2026bridging,
    title={Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations},
    author={Haosen Shi and Jianda Chen and Sinno Jialin Pan},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=jA3KmR18S7}
}
```