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
import sys
import os

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
    fixed_sl_path: str | None = None,
    save_path: str | None = None,
    data_path: str | None = None,
    return_figs: bool = False,
    fixed_sl_ratio: float = 0.95,
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

    # Parse optional stoploss_mask column (backward compat with old CSVs)
    has_stoploss = "stoploss_mask" in df_act.columns

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

    # ── Figure ───────────────────────────────────────────────────────────
    if return_figs:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        axes = [ax1, ax2, ax3]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
        fig.suptitle("Backtesting Results", fontsize=15, fontweight="bold")

    # ── Row 1: Portfolio Value ───────────────────────────────────────────
    ax = axes[0]
    fixed_sl_pct_str = f"{fixed_sl_ratio:.0%}"
    ax.plot(timestamps_acct, total_assets, color="steelblue", linewidth=1.5, label="PPO Agent")
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
    ax.grid(True, alpha=0.3)

    # Overlay fixed stop-loss strategy if provided
    if fixed_sl_path and os.path.exists(fixed_sl_path):
        df_fsl = pd.read_csv(fixed_sl_path, parse_dates=["timestamp"])
        ax.plot(
            df_fsl["timestamp"], df_fsl["total_assets"],
            color="purple", linewidth=1.5, linestyle=":",
            label=f"PPO + Fixed SL ({fixed_sl_pct_str})",
        )

    # Overlay buy-and-hold baseline if provided
    if baseline_path and os.path.exists(baseline_path):
        df_bl = pd.read_csv(baseline_path, parse_dates=["date"])
        ax.plot(
            df_bl["date"], df_bl["total_assets"],
            color="darkorange", linewidth=1.5, linestyle="--",
            label="Buy & Hold Baseline",
        )

    # Initial capital line (plotted last for legend ordering)
    ax.axhline(initial_value, color="gray", linestyle="--", linewidth=0.8, label=f"Initial (${initial_value:,.0f})")

    ax.legend(loc="upper left", fontsize=9)

    # ── Row 2: Max Drawdown ──────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(timestamps_acct, 0, drawdown_pct, color="crimson", alpha=0.35)
    ax.plot(timestamps_acct, drawdown_pct, color="darkred", linewidth=1.0, label="PPO Agent")
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

    # Overlay fixed stop-loss drawdown if provided
    if fixed_sl_path and os.path.exists(fixed_sl_path):
        df_fsl = pd.read_csv(fixed_sl_path, parse_dates=["timestamp"])
        fsl_max = df_fsl["total_assets"].cummax()
        fsl_dd = ((df_fsl["total_assets"] - fsl_max) / fsl_max) * 100
        ax.plot(
            df_fsl["timestamp"], fsl_dd,
            color="indigo", linewidth=1.0, linestyle=":",
            label=f"PPO + Fixed SL ({fixed_sl_pct_str})"
        )

    # Overlay buy-and-hold baseline drawdown if provided
    if baseline_path and os.path.exists(baseline_path):
        df_bl = pd.read_csv(baseline_path, parse_dates=["date"])
        bl_max = df_bl["total_assets"].cummax()
        bl_dd = ((df_bl["total_assets"] - bl_max) / bl_max) * 100
        ax.plot(
            df_bl["date"], bl_dd,
            color="darkorange", linewidth=1.0, linestyle="--",
            label="Buy & Hold Baseline"
        )

    ax.legend(loc="lower left", fontsize=9)

    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown  —  Max: {max_dd:.2f}%", fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Row 3: Close Price with Buy/Sell Arrows ──────────────────────────
    ax = axes[2]
    ts_act = df_act["timestamp"]

    if data_path and os.path.exists(data_path):
        # ── Ticker-level close prices with per-ticker buy/sell arrows ──
        df_data = pd.read_csv(data_path, parse_dates=["date"])
        tickers = sorted(df_data["tic"].unique())
        price_pivot = df_data.pivot_table(index="date", columns="tic", values="close")
        price_pivot = price_pivot[tickers]  # enforce sorted order

        # Normalise timestamps to tz-naive so both sides match for reindex
        price_pivot.index = pd.to_datetime(price_pivot.index, utc=True).tz_localize(None)
        ts_act_naive = pd.to_datetime(ts_act, utc=True).dt.tz_localize(None)

        # Plot each ticker's close price
        colors = plt.cm.tab10.colors
        for idx, tic in enumerate(tickers):
            color = colors[idx % len(colors)]
            ax.plot(
                price_pivot.index, price_pivot[tic],
                linewidth=1.0, color=color, label=tic.upper(),
            )

        # Decompose per-step transactions into per-ticker arrays
        trans_matrix = np.vstack(df_act["transactions_parsed"].values)  # (steps, n_tickers)

        # Parse stoploss mask if available
        if has_stoploss:
            sl_matrix = np.vstack(
                df_act["stoploss_mask"].apply(_parse_array_str).values
            )  # (steps, n_tickers)
        else:
            sl_matrix = np.zeros_like(trans_matrix)

        for idx, tic in enumerate(tickers):
            color = colors[idx % len(colors)]
            per_tic_trans = trans_matrix[:, idx]
            per_tic_sl = sl_matrix[:, idx] > 0.5  # boolean mask

            tic_buy_mask = per_tic_trans > 0
            tic_normal_sell_mask = (per_tic_trans < 0) & ~per_tic_sl
            tic_sl_sell_mask = (per_tic_trans < 0) & per_tic_sl

            # Look up the close price at each action timestamp for this ticker
            tic_prices_at_action = price_pivot.reindex(ts_act_naive)[tic].values

            if tic_buy_mask.any():
                ax.scatter(
                    ts_act_naive[tic_buy_mask], tic_prices_at_action[tic_buy_mask],
                    marker="^", s=50, color=color, edgecolors="darkgreen",
                    linewidths=0.6, zorder=5,
                    label="Buy" if idx == 0 else "",
                )
            if tic_normal_sell_mask.any():
                ax.scatter(
                    ts_act_naive[tic_normal_sell_mask], tic_prices_at_action[tic_normal_sell_mask],
                    marker="v", s=50, color=color, edgecolors="darkred",
                    linewidths=0.6, zorder=5,
                    label="Sell" if idx == 0 else "",
                )
            if tic_sl_sell_mask.any():
                ax.scatter(
                    ts_act_naive[tic_sl_sell_mask], tic_prices_at_action[tic_sl_sell_mask],
                    marker="X", s=70, color="red", edgecolors="black",
                    linewidths=0.8, zorder=6,
                    label="Stop Loss" if idx == 0 else "",
                )

        ax.set_ylabel("Close Price ($)")
        ax.set_xlabel("Date")
        ax.set_title("Ticker Prices with Buy / Sell Signals", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        # ── Fallback: market exposure chart (original behaviour) ──
        asset_value = df_acct["asset_value"]
        ax.plot(timestamps_acct, asset_value, color="slategray", linewidth=1.0, label="Asset Value (market exposure)")

        # Compute net-level stoploss for fallback mode
        if has_stoploss:
            net_sl = df_act["stoploss_mask"].apply(
                lambda s: _parse_array_str(s).max() > 0.5
            )
        else:
            net_sl = pd.Series([False] * len(df_act))

        normal_sell_mask = sell_mask & ~net_sl
        sl_sell_mask = sell_mask & net_sl

        if buy_mask.any():
            buy_ts = ts_act[buy_mask]
            buy_vals = df_acct.set_index("timestamp").reindex(buy_ts)["asset_value"].values
            ax.scatter(
                buy_ts, buy_vals,
                marker="^", s=50, color="limegreen", edgecolors="darkgreen",
                linewidths=0.6, zorder=5, label="Net Buy",
            )

        if normal_sell_mask.any():
            sell_ts = ts_act[normal_sell_mask]
            sell_vals = df_acct.set_index("timestamp").reindex(sell_ts)["asset_value"].values
            ax.scatter(
                sell_ts, sell_vals,
                marker="v", s=50, color="tomato", edgecolors="darkred",
                linewidths=0.6, zorder=5, label="Net Sell",
            )

        if sl_sell_mask.any():
            sl_ts = ts_act[sl_sell_mask]
            sl_vals = df_acct.set_index("timestamp").reindex(sl_ts)["asset_value"].values
            ax.scatter(
                sl_ts, sl_vals,
                marker="X", s=70, color="red", edgecolors="black",
                linewidths=0.8, zorder=6, label="Stop Loss",
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

    if return_figs:
        fig1.autofmt_xdate(rotation=30)
        fig2.autofmt_xdate(rotation=30)
        fig3.autofmt_xdate(rotation=30)
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        return [fig1, fig2, fig3]

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
    parser.add_argument(
        "--fixed_sl_path", type=str, default=None,
        help="Path to fixed_stoploss_account_history.csv for overlay comparison",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to the original preprocessed CSV (with tic/close columns) for per-ticker price chart",
    )
    args = parser.parse_args()
    plot_backtest(args.account_path, args.action_path, args.baseline_path, args.fixed_sl_path, args.save, args.data_path)


if __name__ == "__main__":
    main()
