"""
Tool-1: Training / Learning Curve Visualizer

Reads the training-log CSV produced by RewardLoggingCallback (in train_ppo.py)
and plots the RL learning curve so you can quickly check whether the agent
is actually learning (reward trending up) or something is broken.

Usage:
    uv run python -m src.plot_training_curve --log_path trained_models/my_ppo_bot_training_log.csv
"""

import argparse
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(log_path: str, rolling_window: int = 5, save_path: str | None = None, return_figs: bool = False):
    if not os.path.exists(log_path):
        print(
            f"ERROR: Training log not found at '{log_path}'.\n"
            "This file is created automatically during training by RewardLoggingCallback.\n"
            "Please retrain the model with the updated train_ppo.py to generate it."
        )
        sys.exit(1)

    df = pd.read_csv(log_path)
    if df.empty:
        print("ERROR: Training log CSV is empty — no rollouts were recorded.")
        sys.exit(1)

    ts = df["timesteps"]

    # ── Figure layout ────────────────────────────────────────────────────
    if return_figs:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        axes = [ax1, ax2, ax3]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("PPO Training — Learning Curve", fontsize=15, fontweight="bold")

    # ── Row 1: Episode Reward Mean ───────────────────────────────────────
    ax = axes[0]
    rew = df["ep_rew_mean"]
    ax.scatter(ts, rew, alpha=0.25, s=12, color="steelblue", label="per-rollout")
    if len(rew) >= rolling_window:
        smoothed = rew.rolling(rolling_window, min_periods=1).mean()
        ax.plot(ts, smoothed, color="navy", linewidth=2, label=f"rolling mean (w={rolling_window})")
    ax.set_ylabel("Episode Reward (mean)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Row 2: Policy & Value Loss ───────────────────────────────────────
    ax = axes[1]
    if "policy_loss" in df.columns:
        ax.plot(ts, df["policy_loss"], color="coral", linewidth=1.2, label="Policy loss")
    if "value_loss" in df.columns:
        ax.plot(ts, df["value_loss"], color="mediumpurple", linewidth=1.2, label="Value loss")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Row 3: Entropy & KL / clip ───────────────────────────────────────
    ax = axes[2]
    if "entropy_loss" in df.columns:
        ax.plot(ts, df["entropy_loss"], color="seagreen", linewidth=1.2, label="Entropy loss")
    if "approx_kl" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(ts, df["approx_kl"], color="orange", linewidth=1.0, alpha=0.7, label="Approx KL")
        ax2.set_ylabel("Approx KL", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Entropy Loss")
    ax.set_xlabel("Timesteps")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    if return_figs:
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        return [fig1, fig2, fig3]

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training learning curve")
    parser.add_argument(
        "--log_path", type=str, required=True,
        help="Path to the training_log.csv produced by RewardLoggingCallback",
    )
    parser.add_argument(
        "--rolling_window", type=int, default=5,
        help="Window size for rolling average smoothing (default: 5)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="If set, save figure to this path instead of displaying it",
    )
    args = parser.parse_args()
    plot_training_curve(args.log_path, args.rolling_window, args.save)


if __name__ == "__main__":
    main()
