"""
Result visualization for AMAPPO / AMAPPOv2 / MAPPO comparison.

Usage:
    # Default: all available algorithms
    python experiments/plot_results.py --output_dir figures

    # Plot specific algorithms only
    python experiments/plot_results.py --algos amappo_v2 --output_dir figures
    python experiments/plot_results.py --algos amappo_v2,mappo --output_dir figures

    # Specify directories explicitly
    python experiments/plot_results.py \
        --amappo_dir runs/amappo \
        --amappo_v2_dir runs/amappo_v2 \
        --mappo_dir  runs/mappo \
        --output_dir figures

The script reads TensorBoard event logs and produces:
  1. Convergence curves: reward vs episode (mean ± std across seeds)
  2. Cost convergence curves
  3. Bar chart: last-100-episode average for each algorithm
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
# Plot helpers — parameterized for N algorithms
# ---------------------------------------------------------------------------

# Default algorithm registry: (label, color)
ALGO_STYLE = {
    "amappo":    ("AMAPPO",   "tab:blue"),
    "amappo_v2": ("AMAPPOv2", "tab:green"),
    "mappo":     ("MAPPO",    "tab:orange"),
}


def plot_convergence(
    algo_data: Dict[str, Optional[np.ndarray]],
    output_path: str,
    title: str = "Training Convergence",
    ylabel: str = "Episode Reward",
):
    """
    Line plot of mean ± std across seeds for multiple algorithms.

    algo_data: {algo_key: (n_seeds, T) array or None}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x_label = "Episode (×log_interval)"

    has_data = False
    for algo_key, values in algo_data.items():
        if values is None or values.shape[0] == 0:
            continue
        label, color = ALGO_STYLE.get(algo_key, (algo_key, None))
        mean = values.mean(axis=0)
        std  = values.std(axis=0)
        xs   = np.arange(len(mean))
        ax.plot(xs, mean, label=label, color=color)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.25, color=color)
        has_data = True

    if not has_data:
        plt.close(fig)
        print(f"[plot] No data to plot for {output_path}, skipped.")
        return

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
    algo_last100: Dict[str, float],
    output_path: str,
    metric_name: str = "Avg Episode Reward (last 100 eps)",
):
    """Bar chart comparing final performance across multiple algorithms."""
    import matplotlib.pyplot as plt

    # Filter out zero-value entries that mean "no data"
    filtered = {k: v for k, v in algo_last100.items() if v != 0.0}
    if not filtered:
        print(f"[plot] No data for bar chart, skipped.")
        return

    algos  = []
    values = []
    colors = []
    for algo_key, val in filtered.items():
        label, color = ALGO_STYLE.get(algo_key, (algo_key, None))
        algos.append(label)
        values.append(val)
        colors.append(color)

    width = min(0.4, 1.6 / len(algos))
    fig, ax = plt.subplots(figsize=(max(5, 2 * len(algos)), 4))
    bars = ax.bar(algos, values, color=colors, width=width)

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
    parser = argparse.ArgumentParser(description="Plot AMAPPO / AMAPPOv2 / MAPPO results")
    parser.add_argument("--log_dir",       type=str, default="runs",
                        help="Base TensorBoard log directory (contains algo subdirs)")
    parser.add_argument("--algos",         type=str, default=None,
                        help="Comma-separated list of algorithms to plot "
                             "(e.g. amappo,amappo_v2,mappo). "
                             "Default: all available algorithms")
    parser.add_argument("--amappo_dir",    type=str, default=None,
                        help="Override: explicit amappo TensorBoard directory")
    parser.add_argument("--amappo_v2_dir", type=str, default=None,
                        help="Override: explicit amappo_v2 TensorBoard directory")
    parser.add_argument("--mappo_dir",     type=str, default=None,
                        help="Override: explicit mappo TensorBoard directory")
    parser.add_argument("--output_dir",    type=str, default="figures",
                        help="Directory to save output figures")
    parser.add_argument("--tag",           type=str, default="episode/reward",
                        help="TensorBoard scalar tag to plot")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # All known algorithms
    all_algo_keys = ["amappo", "amappo_v2", "mappo"]

    # Determine which algorithms to plot
    if args.algos is not None:
        selected = [a.strip() for a in args.algos.split(",")]
        unknown = set(selected) - set(all_algo_keys)
        if unknown:
            parser = parse_args()
            parser.error(f"Unknown algorithm(s): {unknown}. Available: {all_algo_keys}")
    else:
        selected = all_algo_keys

    # Resolve directories (only for selected algorithms)
    default_dirs = {
        "amappo":    args.amappo_dir    or os.path.join(args.log_dir, "amappo"),
        "amappo_v2": args.amappo_v2_dir or os.path.join(args.log_dir, "amappo_v2"),
        "mappo":     args.mappo_dir     or os.path.join(args.log_dir, "mappo"),
    }
    algo_dirs = {k: default_dirs[k] for k in selected}

    for algo_key, d in algo_dirs.items():
        label = ALGO_STYLE.get(algo_key, (algo_key,))[0]
        print(f"[plot] Reading {label} from: {d}")

    # Collect reward data
    reward_stacked: Dict[str, Optional[np.ndarray]] = {}
    for algo_key, base_dir in algo_dirs.items():
        if not os.path.isdir(base_dir):
            print(f"[plot] Directory not found, skipping {algo_key}: {base_dir}")
            reward_stacked[algo_key] = None
            continue
        runs = collect_runs(base_dir, args.tag)
        reward_stacked[algo_key] = align_and_stack(runs)

    # Build title from available algorithms
    active_labels = [
        ALGO_STYLE[k][0] for k, v in reward_stacked.items() if v is not None
    ]
    title_suffix = " vs ".join(active_labels) if active_labels else "No Data"

    # Convergence curve — reward
    plot_convergence(
        reward_stacked,
        output_path=os.path.join(args.output_dir, "convergence_reward.png"),
        title=f"{title_suffix} — Training Convergence",
        ylabel="Episode Reward",
    )

    # Bar chart: last 100 episodes
    def last100_mean(stacked: Optional[np.ndarray]) -> float:
        if stacked is None:
            return 0.0
        return float(stacked[:, -100:].mean())

    algo_last100 = {k: last100_mean(v) for k, v in reward_stacked.items()}

    for algo_key, val in algo_last100.items():
        label = ALGO_STYLE.get(algo_key, (algo_key,))[0]
        print(f"[plot] {label} last-100 avg reward: {val:.4f}")

    plot_bar_comparison(
        algo_last100,
        output_path=os.path.join(args.output_dir, "bar_comparison.png"),
    )

    # Cost convergence
    cost_tag = "episode/cost"
    cost_stacked: Dict[str, Optional[np.ndarray]] = {}
    for algo_key, base_dir in algo_dirs.items():
        if not os.path.isdir(base_dir):
            cost_stacked[algo_key] = None
            continue
        cost_runs = collect_runs(base_dir, cost_tag)
        cost_stacked[algo_key] = align_and_stack(cost_runs)

    has_cost = any(v is not None for v in cost_stacked.values())
    if has_cost:
        plot_convergence(
            cost_stacked,
            output_path=os.path.join(args.output_dir, "convergence_cost.png"),
            title=f"{title_suffix} — System Cost",
            ylabel="System Cost",
        )

    print("[plot] Done.")


if __name__ == "__main__":
    main()
