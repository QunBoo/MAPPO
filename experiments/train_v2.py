"""
AMAPPOv2 independent training entry point.

Usage:
    python experiments/train_v2.py --epochs 1500 --seed 42 --device cuda

This script only imports v2 module paths, fully isolated from train.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from algorithms.amappo_v2 import AMAPPOv2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train AMAPPOv2")
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--gamma",           type=float, default=None)
    parser.add_argument("--gae_lambda",      type=float, default=None)
    parser.add_argument("--eps_clip",        type=float, default=None)
    parser.add_argument("--mini_batch_size", type=int,   default=None)
    parser.add_argument("--max_grad_norm",   type=float, default=None)
    parser.add_argument("--gru_hidden",      type=int,   default=None)
    parser.add_argument("--log_interval",    type=int,   default=None)
    parser.add_argument("--save_interval",   type=int,   default=None)
    parser.add_argument("--log_dir",         type=str,   default=None)
    parser.add_argument("--checkpoint_dir",  type=str,   default=None)
    parser.add_argument("--device",          type=str,   default=None, choices=["cpu", "cuda"])
    parser.add_argument("--max_steps",       type=int,   default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = Config()
    cfg.algo = "amappo_v2"

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

    set_seed(cfg.seed)

    print(f"[train_v2.py] algo=amappo_v2  epochs={cfg.epochs}  seed={cfg.seed}")
    print(f"              device={cfg.device}  lr={cfg.lr}  mini_batch={cfg.mini_batch_size}")

    start_time = time.time()
    trainer = AMAPPOv2Trainer(cfg)
    trainer.train()

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"[train_v2.py] Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    main()
