"""
Tool-2: Backtesting Results Visualizer

Reads the account-history and action-history CSVs produced by backtest.py
and plots:
  1. Portfolio total assets over time
  2. Max drawdown over time
  3. Close prices with buy/sell arrows

Usage:
    uv run python -m src.plot_backtest \
        --account_path results/my_ppo_bot_account_history.csv \
        --action_path  results/my_ppo_bot_action_history.csv
"""

import argparse
import ast
import sys
import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _parse_array_str(s: str) -> np.ndarray:
    """Parse a stringified numpy array like '[ 628.  -0. 152.]' into a real array."""
    s = s.strip()
    # Normalise whitespace inside brackets so it becomes comma-separated
    inner = s.strip("[]")
    tokens = inner.split()
    return np.array([float(t) for t in tokens])


def plot_backtest(
    account_path: str,
    action_path: str,
    baseline_path: str | None = None,
    save_path: str | None = None,
):
    # ── Load data ────────────────────────────────────────────────────────
    if not os.path.exists(account_path):
        print(f"ERROR: account history not found at '{account_path}'")
        sys.exit(1)
    if not os.path.exists(action_path):
        print(f"ERROR: action history not found at '{action_path}'")
        sys.exit(1)

    df_acct = pd.read_csv(account_path, parse_dates=["timestamp"])
    df_act = pd.read_csv(action_path, parse_dates=["timestamp"])

    # Parse the stringified arrays in transactions column
    df_act["transactions_parsed"] = df_act["transactions"].apply(_parse_array_str)

    timestamps_acct = df_acct["timestamp"]
    total_assets = df_acct["total_assets"]
    initial_value = total_assets.iloc[0]

    # ── Compute drawdown series ──────────────────────────────────────────
    running_max = total_assets.cummax()
    drawdown_pct = ((total_assets - running_max) / running_max) * 100  # negative %

    # ── Classify each step as net-buy / net-sell / hold ───────────────────
    net_positions = df_act["transactions_parsed"].apply(lambda arr: arr.sum())
    buy_mask = net_positions > 0
    sell_mask = net_positions < 0

    # ── Figure: 3 rows ───────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Backtesting Results", fontsize=15, fontweight="bold")

    # ── Row 1: Portfolio Value ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(timestamps_acct, total_assets, color="steelblue", linewidth=1.5, label="Total Assets")
    ax.axhline(initial_value, color="gray", linestyle="--", linewidth=0.8, label=f"Initial (${initial_value:,.0f})")
    ax.fill_between(
        timestamps_acct, initial_value, total_assets,
        where=(total_assets >= initial_value), alpha=0.15, color="green", interpolate=True,
    )
    ax.fill_between(
        timestamps_acct, initial_value, total_assets,
        where=(total_assets < initial_value), alpha=0.15, color="red", interpolate=True,
    )
    final_val = total_assets.iloc[-1]
    ret_pct = ((final_val - initial_value) / initial_value) * 100
    ax.set_title(
        f"Portfolio Value  —  Final: ${final_val:,.0f}  |  Return: {ret_pct:+.2f}%",
        fontsize=11,
    )
    ax.set_ylabel("Total Assets ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Overlay buy-and-hold baseline if provided
    if baseline_path and os.path.exists(baseline_path):
        df_bl = pd.read_csv(baseline_path, parse_dates=["date"])
        ax.plot(
            df_bl["date"], df_bl["total_assets"],
            color="darkorange", linewidth=1.5, linestyle="--",
            label="Buy & Hold Baseline",
        )
        ax.legend(loc="upper left", fontsize=9)

    # ── Row 2: Max Drawdown ──────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(timestamps_acct, 0, drawdown_pct, color="crimson", alpha=0.35)
    ax.plot(timestamps_acct, drawdown_pct, color="darkred", linewidth=1.0)
    max_dd = drawdown_pct.min()
    max_dd_idx = drawdown_pct.idxmin()
    ax.annotate(
        f"Max DD: {max_dd:.2f}%",
        xy=(timestamps_acct.iloc[max_dd_idx], max_dd),
        xytext=(15, -15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
    )
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown  —  Max: {max_dd:.2f}%", fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Row 3: Close Price with Buy/Sell Arrows ──────────────────────────
    ax = axes[2]

    # Parse actions column to approximate close price (actions are dollar amounts)
    # Use the account-level data to derive an implicit price index
    # Better: use asset_value / estimated shares, but simplest proxy is the
    # asset_value trend itself as a "market exposure" indicator.
    # Since we don't have raw close prices in the results CSVs, we plot
    # asset_value (the market-exposure component) as the price proxy.
    asset_value = df_acct["asset_value"]
    ax.plot(timestamps_acct, asset_value, color="slategray", linewidth=1.0, label="Asset Value (market exposure)")

    # Overlay buy arrows (green ↑) and sell arrows (red ↓)
    ts_act = df_act["timestamp"]

    if buy_mask.any():
        buy_ts = ts_act[buy_mask]
        # Find corresponding asset_value at those timestamps
        buy_vals = df_acct.set_index("timestamp").reindex(buy_ts)["asset_value"].values
        ax.scatter(
            buy_ts, buy_vals,
            marker="^", s=50, color="limegreen", edgecolors="darkgreen",
            linewidths=0.6, zorder=5, label="Net Buy",
        )

    if sell_mask.any():
        sell_ts = ts_act[sell_mask]
        sell_vals = df_acct.set_index("timestamp").reindex(sell_ts)["asset_value"].values
        ax.scatter(
            sell_ts, sell_vals,
            marker="v", s=50, color="tomato", edgecolors="darkred",
            linewidths=0.6, zorder=5, label="Net Sell",
        )

    ax.set_ylabel("Asset Value ($)")
    ax.set_xlabel("Date")
    ax.set_title("Market Exposure with Buy / Sell Signals", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Nice date formatting
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        a.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot backtesting results with buy/sell arrows")
    parser.add_argument(
        "--account_path", type=str, required=True,
        help="Path to the account_history CSV from backtest.py",
    )
    parser.add_argument(
        "--action_path", type=str, required=True,
        help="Path to the action_history CSV from backtest.py",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="If set, save figure to this path instead of displaying it",
    )
    parser.add_argument(
        "--baseline_path", type=str, default=None,
        help="Path to baseline_buy_and_hold_account_history.csv for overlay comparison",
    )
    args = parser.parse_args()
    plot_backtest(args.account_path, args.action_path, args.baseline_path, args.save)


if __name__ == "__main__":
    main()
