"""Analyse hparam_results.csv and generate Pareto front + metrics plots.

Axes (Pareto plot):
  X — Average PPO Return (higher is better → maximise)
  Y — Average Max Drawdown
       (less negative / closer to 0 is better → maximise, i.e. minimise |DD|)

Each point represents one `reward_weight_drawdown` setting, averaged across
all tickers × rolling windows.  The Pareto front connects the non-dominated
points.

A second figure compares Sharpe and Sortino ratios across PPO, Fixed-SL,
and Buy-and-Hold strategies for each weight setting.

Usage:
    uv run python -m src.analyse_pareto
    uv run python -m src.analyse_pareto --input path/to/results.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and validate expected columns."""
    df = pd.read_csv(path)
    required = {
        "reward_weight_drawdown",
        "ppo_return", "ppo_max_dd", "ppo_sharpe", "ppo_sortino",
        "fsl_return", "fsl_max_dd", "fsl_sharpe", "fsl_sortino",
        "bh_return", "bh_max_dd", "bh_sharpe", "bh_sortino",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def aggregate_by_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across all tickers × windows for each reward_weight_drawdown.

    Returns a DataFrame with columns for return/DD/Sharpe/Sortino across
    all three strategies (PPO, FSL, B&H).
    """
    grouped = df.groupby("reward_weight_drawdown").agg(
        avg_return=("ppo_return", "mean"),
        avg_max_dd=("ppo_max_dd", "mean"),
        worst_max_dd=("ppo_max_dd", "min"),   # most negative = worst
        median_return=("ppo_return", "median"),
        count=("ppo_return", "size"),
        # PPO Sharpe & Sortino
        avg_ppo_sharpe=("ppo_sharpe", "mean"),
        avg_ppo_sortino=("ppo_sortino", "mean"),
        # Fixed Stop-Loss Sharpe & Sortino
        avg_fsl_sharpe=("fsl_sharpe", "mean"),
        avg_fsl_sortino=("fsl_sortino", "mean"),
        # Buy-and-Hold Sharpe & Sortino
        avg_bh_sharpe=("bh_sharpe", "mean"),
        avg_bh_sortino=("bh_sortino", "mean"),
        # Strategy returns for comparison
        avg_fsl_return=("fsl_return", "mean"),
        avg_bh_return=("bh_return", "mean"),
        avg_fsl_max_dd=("fsl_max_dd", "mean"),
        avg_bh_max_dd=("bh_max_dd", "mean"),
    ).reset_index()
    return grouped


def pareto_front(avg_returns: np.ndarray, avg_drawdowns: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-optimal points.

    We are maximising both objectives:
      - avg_return  (higher is better)
      - avg_max_dd  (less negative / closer to 0 is better)
    """
    n = len(avg_returns)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= in both objectives and > in at least one
            if (avg_returns[j] >= avg_returns[i] and
                avg_drawdowns[j] >= avg_drawdowns[i] and
                (avg_returns[j] > avg_returns[i] or
                 avg_drawdowns[j] > avg_drawdowns[i])):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]


def plot_pareto(agg: pd.DataFrame, output_path: str) -> None:
    """Generate and save the Pareto front plot."""
    avg_ret = agg["avg_return"].values
    avg_dd = agg["avg_max_dd"].values
    weights = agg["reward_weight_drawdown"].values

    # Compute Pareto front indices
    pf_idx = pareto_front(avg_ret, avg_dd)

    # Sort Pareto front by avg_return for a clean connecting line
    pf_sorted = pf_idx[np.argsort(avg_ret[pf_idx])]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # All points
    scatter = ax.scatter(
        avg_ret * 100,
        avg_dd * 100,
        c=weights,
        cmap="viridis",
        s=120,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        alpha=0.85,
    )

    # Pareto front line
    ax.plot(
        avg_ret[pf_sorted] * 100,
        avg_dd[pf_sorted] * 100,
        color="red",
        linewidth=2,
        linestyle="--",
        label="Pareto Front",
        zorder=2,
    )

    # Highlight Pareto-optimal points
    ax.scatter(
        avg_ret[pf_sorted] * 100,
        avg_dd[pf_sorted] * 100,
        facecolors="none",
        edgecolors="red",
        s=200,
        linewidths=2,
        zorder=4,
        label="Pareto-optimal",
    )

    # Annotate each point with its weight value
    for i, w in enumerate(weights):
        ax.annotate(
            f"{w:.2f}",
            (avg_ret[i] * 100, avg_dd[i] * 100),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
            color="0.3",
        )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("reward_weight_drawdown", fontsize=11)

    # Labels and title
    ax.set_xlabel("Average PPO Return (%)", fontsize=13)
    ax.set_ylabel("Average Max Drawdown (%)", fontsize=13)
    ax.set_title(
        "Pareto Front: Average Profit vs Max Drawdown\n"
        "(reward_weight_pnl = 1.0, varying reward_weight_drawdown)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add an arrow indicating the "ideal" direction
    ax.annotate(
        "Ideal ↗",
        xy=(0.92, 0.92),
        xycoords="axes fraction",
        fontsize=11,
        fontweight="bold",
        color="green",
        alpha=0.7,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


def plot_sharpe_comparison(agg: pd.DataFrame, output_path: str) -> None:
    """Generate a grouped bar chart comparing Sharpe ratios
    across PPO, Fixed-SL, and Buy-and-Hold strategies by weight.
    """
    weights = agg["reward_weight_drawdown"].values
    x = np.arange(len(weights))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(
        x - bar_width, agg["avg_ppo_sharpe"].values,
        bar_width, label="PPO", color="#3b82f6", edgecolor="white",
    )
    ax.bar(
        x, agg["avg_fsl_sharpe"].values,
        bar_width, label="Fixed SL", color="#f97316", edgecolor="white",
    )
    ax.bar(
        x + bar_width, agg["avg_bh_sharpe"].values,
        bar_width, label="Buy & Hold", color="#22c55e", edgecolor="white",
    )
    ax.set_xlabel("reward_weight_drawdown", fontsize=12)
    ax.set_ylabel("Average Sharpe Ratio", fontsize=12)
    ax.set_title(
        "Sharpe Ratio by Strategy & Weight\n"
        "PPO vs Fixed-SL vs Buy-and-Hold",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:.2f}" for w in weights], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="-")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Sharpe comparison saved to: {output_path}")
    plt.close(fig)


def plot_sortino_comparison(agg: pd.DataFrame, output_path: str) -> None:
    """Generate a grouped bar chart comparing Sortino ratios
    across PPO, Fixed-SL, and Buy-and-Hold strategies by weight.
    """
    weights = agg["reward_weight_drawdown"].values
    x = np.arange(len(weights))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(
        x - bar_width, agg["avg_ppo_sortino"].values,
        bar_width, label="PPO", color="#3b82f6", edgecolor="white",
    )
    ax.bar(
        x, agg["avg_fsl_sortino"].values,
        bar_width, label="Fixed SL", color="#f97316", edgecolor="white",
    )
    ax.bar(
        x + bar_width, agg["avg_bh_sortino"].values,
        bar_width, label="Buy & Hold", color="#22c55e", edgecolor="white",
    )
    ax.set_xlabel("reward_weight_drawdown", fontsize=12)
    ax.set_ylabel("Average Sortino Ratio", fontsize=12)
    ax.set_title(
        "Sortino Ratio by Strategy & Weight\n"
        "PPO vs Fixed-SL vs Buy-and-Hold",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:.2f}" for w in weights], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="-")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Sortino comparison saved to: {output_path}")
    plt.close(fig)


def print_summary(agg: pd.DataFrame, pf_idx: np.ndarray) -> None:
    """Print a summary table to stdout."""
    print("\n" + "=" * 120)
    print("AGGREGATED RESULTS BY reward_weight_drawdown")
    print("=" * 120)

    # Header row 1: Return & Drawdown
    print(
        f"{'Weight':>8s}  {'Avg Ret%':>10s}  {'Avg DD%':>9s}  "
        f"{'WorstDD%':>10s}  {'MedRet%':>10s}  "
        f"{'PPO Shp':>8s}  {'PPO Sort':>9s}  "
        f"{'FSL Shp':>8s}  {'FSL Sort':>9s}  "
        f"{'B&H Shp':>8s}  {'B&H Sort':>9s}  "
        f"{'N':>4s}  {'PF':>3s}"
    )
    print("-" * 120)
    for i, row in agg.iterrows():
        is_pf = " ★" if i in pf_idx else ""
        print(
            f"{row['reward_weight_drawdown']:8.2f}  "
            f"{row['avg_return']*100:10.4f}  "
            f"{row['avg_max_dd']*100:9.4f}  "
            f"{row['worst_max_dd']*100:10.4f}  "
            f"{row['median_return']*100:10.4f}  "
            f"{row['avg_ppo_sharpe']:8.4f}  "
            f"{row['avg_ppo_sortino']:9.4f}  "
            f"{row['avg_fsl_sharpe']:8.4f}  "
            f"{row['avg_fsl_sortino']:9.4f}  "
            f"{row['avg_bh_sharpe']:8.4f}  "
            f"{row['avg_bh_sortino']:9.4f}  "
            f"{row['count']:4.0f}{is_pf}"
        )
    print("=" * 120)

    print("\nPareto-optimal weights (non-dominated on avg return × avg max DD):")
    for idx in pf_idx:
        row = agg.iloc[idx]
        print(
            f"  weight={row['reward_weight_drawdown']:.2f}  "
            f"ret={row['avg_return']*100:.4f}%  "
            f"dd={row['avg_max_dd']*100:.4f}%  "
            f"sharpe={row['avg_ppo_sharpe']:.4f}  "
            f"sortino={row['avg_ppo_sortino']:.4f}"
        )

    # Cross-strategy comparison summary
    print("\n" + "-" * 80)
    print("CROSS-STRATEGY COMPARISON (averaged across all weights)")
    print("-" * 80)
    for metric, label in [
        ("avg_ppo_sharpe", "PPO Sharpe"),
        ("avg_fsl_sharpe", "Fixed-SL Sharpe"),
        ("avg_bh_sharpe", "Buy-and-Hold Sharpe"),
        ("avg_ppo_sortino", "PPO Sortino"),
        ("avg_fsl_sortino", "Fixed-SL Sortino"),
        ("avg_bh_sortino", "Buy-and-Hold Sortino"),
    ]:
        print(f"  {label:<22s}: {agg[metric].mean():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Pareto front analysis for MORL results.")
    parser.add_argument(
        "--input", type=str, default="hparam_results.csv",
        help="Path to the hyperparameter results CSV.",
    )
    parser.add_argument(
        "--output", type=str, default="pareto_front.png",
        help="Output path for the Pareto front plot.",
    )
    parser.add_argument(
        "--output-sharpe", type=str, default="sharpe_comparison.png",
        help="Output path for the Sharpe ratio comparison plot.",
    )
    parser.add_argument(
        "--output-sortino", type=str, default="sortino_comparison.png",
        help="Output path for the Sortino ratio comparison plot.",
    )
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Unique reward_weight_drawdown values: {sorted(df['reward_weight_drawdown'].unique())}")
    print(f"Unique tickers: {sorted(df['tickers'].unique())}")
    print(f"Test windows: {df.groupby(['test_start','test_end']).ngroups}")

    agg = aggregate_by_weight(df)

    avg_ret = agg["avg_return"].values
    avg_dd = agg["avg_max_dd"].values
    pf_idx = pareto_front(avg_ret, avg_dd)

    print_summary(agg, pf_idx)
    plot_pareto(agg, args.output)
    plot_sharpe_comparison(agg, args.output_sharpe)
    plot_sortino_comparison(agg, args.output_sortino)


if __name__ == "__main__":
    main()
