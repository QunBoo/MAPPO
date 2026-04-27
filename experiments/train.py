"""
Training entry point for AMAPPO / MAPPO.

Usage:
    python experiments/train.py --algo amappo --epochs 1500 --seed 42
    python experiments/train.py --algo mappo  --epochs 1500 --seed 42

Override any Config field with --key value, e.g.:
    python experiments/train.py --algo amappo --epochs 100 --lr 1e-3 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import time
import numpy as np
import torch

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AMAPPO / MAPPO")

    # Core switches
    parser.add_argument("--algo",   type=str,   default="amappo",
                        choices=["amappo", "mappo"],
                        help="Algorithm to run")
    parser.add_argument("--epochs", type=int,   default=None,
                        help="Number of training episodes (overrides config)")
    parser.add_argument("--seed",   type=int,   default=None,
                        help="Random seed (overrides config)")

    # Config overrides (subset of common ones)
    parser.add_argument("--lr",                 type=float, default=None)
    parser.add_argument("--gamma",              type=float, default=None)
    parser.add_argument("--gae_lambda",         type=float, default=None)
    parser.add_argument("--eps_clip",           type=float, default=None)
    parser.add_argument("--mini_batch_size",    type=int,   default=None)
    parser.add_argument("--max_grad_norm",      type=float, default=None)
    parser.add_argument("--gru_hidden",         type=int,   default=None)
    parser.add_argument("--log_interval",       type=int,   default=None)
    parser.add_argument("--save_interval",      type=int,   default=None)
    parser.add_argument("--log_dir",            type=str,   default=None)
    parser.add_argument("--checkpoint_dir",     type=str,   default=None)
    parser.add_argument("--device",             type=str,   default=None,
                        choices=["cpu", "cuda"])
    parser.add_argument("--max_steps",          type=int,   default=None,
                        help="Max steps per episode")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build a Config, applying CLI overrides."""
    cfg = Config()
    cfg.algo = args.algo

    # Apply overrides for fields that are not None
    override_fields = [
        "epochs", "seed", "lr", "gamma", "gae_lambda", "eps_clip",
        "mini_batch_size", "max_grad_norm", "gru_hidden", "log_interval",
        "save_interval", "log_dir", "checkpoint_dir", "device", "max_steps",
    ]
    for field in override_fields:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    cfg.sync_derived_fields()

    return cfg


def set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args   = parse_args()
    config = build_config(args)

    print(f"[train.py] algo={config.algo}  epochs={config.epochs}  seed={config.seed}")
    print(f"           device={config.device}  lr={config.lr}  mini_batch={config.mini_batch_size}")

    set_seed(config.seed)

    start_time = time.time()

    if config.algo == "amappo":
        from algorithms.amappo import AMAPPOTrainer
        trainer = AMAPPOTrainer(config)
    elif config.algo == "mappo":
        from algorithms.mappo import MAPPOTrainer
        trainer = MAPPOTrainer(config)
    else:
        raise ValueError(f"Unknown algo: {config.algo}")

    trainer.train()

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"[train.py] Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    main()
