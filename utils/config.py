from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Environment
    N: int = 100          # number of IoTDs
    M: int = 4            # number of UAVs (agents)
    K: int = 8            # number of LEO satellites
    J: int = 20           # tasks per DAG
    area_size: float = 1000.0   # m
    dt: float = 1.0             # timestep seconds
    max_steps: int = 200        # max steps per episode
    H_uav: float = 50.0         # UAV altitude m
    v_max: float = 30.0         # max UAV speed m/s
    d_min: float = 3.0          # min safe distance m

    # Reward
    eta_t: float = 0.5          # time weight
    eta_e: float = 0.5          # energy weight
    lambda_c: float = 10.0      # constraint penalty

    # Training
    gamma: float = 0.99         # discount factor
    gae_lambda: float = 0.95    # GAE lambda
    eps_clip: float = 0.2       # PPO clip epsilon
    lr: float = 5e-4            # learning rate
    mini_batch_size: int = 128  # mini-batch size
    epochs: int = 1500          # total training episodes
    max_grad_norm: float = 0.5  # gradient clipping

    # Network
    gru_hidden: int = 64        # GRU hidden size
    gnn_out_dim: int = 128      # GNN output dim
    obs_dim: int = 37           # observation dim (per agent)
    action_dim: int = 8         # action dim (per agent)

    # Logging
    log_interval: int = 100     # log every N episodes
    save_interval: int = 100    # checkpoint every N episodes
    log_dir: str = "runs"       # TensorBoard log dir
    checkpoint_dir: str = "checkpoints"

    # Misc
    seed: int = 42
    device: str = "cpu"         # "cpu" or "cuda"
    algo: str = "amappo"        # "amappo" or "mappo"

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)
