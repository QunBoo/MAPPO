"""
Result visualization for AMAPPO vs MAPPO comparison.

Usage:
    # After training with multiple seeds, plot convergence curves:
    python experiments/plot_results.py --log_dir runs --output_dir figures

    # Or point to specific TensorBoard event files:
    python experiments/plot_results.py --amappo_dir runs/amappo --mappo_dir runs/mappo

The script reads TensorBoard event logs and produces:
  1. Convergence curves: reward vs episode (mean ± std across seeds)
  2. Cost bar chart: last-100-episode average for each algorithm
"""

from __future__ import annotations

import os
import sys
import argparse
import glob
from typing import List, Dict, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# TensorBoard log reader
# ---------------------------------------------------------------------------

def read_tb_scalars(log_dir: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a scalar tag from a TensorBoard event directory.

    Returns (steps, values) as 1-D numpy arrays, sorted by step.
    Falls back gracefully if tensorboard is unavailable.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("[plot] tensorboard package not found. Install with: pip install tensorboard")
        return np.array([]), np.array([])

    ea = EventAccumulator(log_dir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if tag not in available_tags:
        print(f"[plot] Tag '{tag}' not found in {log_dir}. Available: {available_tags}")
        return np.array([]), np.array([])

    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    order  = np.argsort(steps)
    return steps[order], values[order]


def collect_runs(base_dir: str, tag: str) -> List[np.ndarray]:
    """
    Collect scalar values from all sub-directories of base_dir (one per seed).
    If base_dir contains event files directly (single-seed run), read it as-is.
    Returns a list of value arrays (may differ in length).
    """
    runs = []
    subdirs = sorted(
        d for d in glob.glob(os.path.join(base_dir, "*"))
        if os.path.isdir(d)
    )
    if not subdirs:
        # No sub-directories — event files live directly in base_dir
        subdirs = [base_dir]

    for d in subdirs:
        _, vals = read_tb_scalars(d, tag)
        if len(vals) > 0:
            runs.append(vals)
    return runs


def align_and_stack(runs: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Truncate all runs to the shortest length and stack as (n_seeds, T).
    Returns None if no runs.
    """
    if not runs:
        return None
    min_len = min(len(r) for r in runs)
    return np.stack([r[:min_len] for r in runs], axis=0)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_convergence(
    amappo_values: Optional[np.ndarray],
    mappo_values:  Optional[np.ndarray],
    output_path: str,
    title: str = "Training Convergence",
    ylabel: str = "Episode Reward",
):
    """
    Line plot of mean ± std across seeds.
    amappo_values / mappo_values: (n_seeds, T) or None
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x_label = "Episode (×log_interval)"

    for label, values, color in [
        ("AMAPPO", amappo_values, "tab:blue"),
        ("MAPPO",  mappo_values,  "tab:orange"),
    ]:
        if values is None or values.shape[0] == 0:
            continue
        mean = values.mean(axis=0)
        std  = values.std(axis=0)
        xs   = np.arange(len(mean))
        ax.plot(xs, mean, label=label, color=color)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.25, color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


def plot_bar_comparison(
    amappo_last100: float,
    mappo_last100: float,
    output_path: str,
    metric_name: str = "Avg Episode Reward (last 100 eps)",
):
    """Bar chart comparing final performance."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    algos  = ["AMAPPO", "MAPPO"]
    values = [amappo_last100, mappo_last100]
    colors = ["tab:blue", "tab:orange"]
    bars   = ax.bar(algos, values, color=colors, width=0.4)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylabel(metric_name)
    ax.set_title("Algorithm Comparison (Final Performance)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot AMAPPO vs MAPPO results")
    parser.add_argument("--log_dir",     type=str, default="runs",
                        help="Base TensorBoard log directory (contains amappo/ and mappo/ subdirs)")
    parser.add_argument("--amappo_dir",  type=str, default=None,
                        help="Override: explicit amappo TensorBoard directory")
    parser.add_argument("--mappo_dir",   type=str, default=None,
                        help="Override: explicit mappo TensorBoard directory")
    parser.add_argument("--output_dir",  type=str, default="figures",
                        help="Directory to save output figures")
    parser.add_argument("--tag",         type=str, default="episode/reward",
                        help="TensorBoard scalar tag to plot")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    amappo_base = args.amappo_dir or os.path.join(args.log_dir, "amappo")
    mappo_base  = args.mappo_dir  or os.path.join(args.log_dir, "mappo")

    print(f"[plot] Reading AMAPPO from: {amappo_base}")
    print(f"[plot] Reading MAPPO  from: {mappo_base}")

    amappo_runs = collect_runs(amappo_base, args.tag)
    mappo_runs  = collect_runs(mappo_base,  args.tag)

    amappo_stacked = align_and_stack(amappo_runs)
    mappo_stacked  = align_and_stack(mappo_runs)

    # Convergence curve
    plot_convergence(
        amappo_stacked, mappo_stacked,
        output_path=os.path.join(args.output_dir, "convergence_reward.png"),
        title="AMAPPO vs MAPPO — Training Convergence",
        ylabel="Episode Reward",
    )

    # Bar chart: last 100 episodes
    def last100_mean(stacked: Optional[np.ndarray]) -> float:
        if stacked is None:
            return 0.0
        return float(stacked[:, -100:].mean())

    amappo_final = last100_mean(amappo_stacked)
    mappo_final  = last100_mean(mappo_stacked)

    print(f"[plot] AMAPPO last-100 avg reward: {amappo_final:.4f}")
    print(f"[plot] MAPPO  last-100 avg reward: {mappo_final:.4f}")

    plot_bar_comparison(
        amappo_final, mappo_final,
        output_path=os.path.join(args.output_dir, "bar_comparison.png"),
    )

    # Also plot cost (negative reward) if tag changes don't already do this
    cost_tag = "episode/cost"
    amappo_cost_runs = collect_runs(amappo_base, cost_tag)
    mappo_cost_runs  = collect_runs(mappo_base,  cost_tag)
    amappo_cost = align_and_stack(amappo_cost_runs)
    mappo_cost  = align_and_stack(mappo_cost_runs)

    if amappo_cost is not None or mappo_cost is not None:
        plot_convergence(
            amappo_cost, mappo_cost,
            output_path=os.path.join(args.output_dir, "convergence_cost.png"),
            title="AMAPPO vs MAPPO — System Cost",
            ylabel="System Cost",
        )

    print("[plot] Done.")


if __name__ == "__main__":
    main()
