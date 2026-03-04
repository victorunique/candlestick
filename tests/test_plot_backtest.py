"""Tests for plot_backtest module – TDD Red phase for ticker-price third graph."""

import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np

from src.plot_backtest import plot_backtest


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_csvs(tmp_path):
    """Create minimal synthetic CSVs (account, action, data) for testing.
    
    Uses timezone-aware timestamps (like real backtest output) to exercise
    the tz normalisation logic.
    """
    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="US/Eastern")

    # --- account_history CSV ---
    acct_rows = []
    for i, d in enumerate(dates):
        acct_rows.append({
            "timestamp": d,
            "cash": 900_000 - i * 10_000,
            "asset_value": 100_000 + i * 15_000,
            "total_assets": 1_000_000 + i * 5_000,
            "reward": 0.01 * i,
        })
    df_acct = pd.DataFrame(acct_rows)
    acct_path = str(tmp_path / "account_history.csv")
    df_acct.to_csv(acct_path, index=False)

    # --- action_history CSV ---
    # transactions array: 2 values per row (one per ticker, sorted)
    act_rows = []
    for i, d in enumerate(dates):
        # Alternate buy / sell pattern per ticker
        t0 = 10 if i % 2 == 0 else -5   # AAPL
        t1 = -3 if i % 2 == 0 else 8    # MSFT
        act_rows.append({
            "timestamp": d,
            "actions": f"[{1000.0} {-500.0}]",
            "transactions": f"[{float(t0)} {float(t1)}]",
        })
    df_act = pd.DataFrame(act_rows)
    act_path = str(tmp_path / "action_history.csv")
    df_act.to_csv(act_path, index=False)

    # --- original data CSV (with close prices per ticker) ---
    data_rows = []
    for d in dates:
        for tic, base in [("AAPL", 150.0), ("MSFT", 380.0)]:
            data_rows.append({
                "date": d,
                "tic": tic,
                "open": base + np.random.uniform(-2, 2),
                "high": base + 3,
                "low": base - 3,
                "close": base + np.random.uniform(-1, 1),
                "volume": 1_000_000,
            })
    df_data = pd.DataFrame(data_rows)
    data_path = str(tmp_path / "data.csv")
    df_data.to_csv(data_path, index=False)
    
    # --- fixed_stoploss_account_history CSV ---
    # same shape as account_history
    fsl_path = str(tmp_path / "fixed_stoploss_account_history.csv")
    df_fsl = df_acct.copy()
    df_fsl["total_assets"] = df_fsl["total_assets"] * 1.05  # slightly higher for visibility
    df_fsl.to_csv(fsl_path, index=False)

    return acct_path, act_path, data_path, fsl_path


# ── Tests ────────────────────────────────────────────────────────────────

def test_plot_backtest_with_data_path_creates_ticker_lines(synthetic_csvs, tmp_path):
    """When data_path is provided, the third axis should have one line per ticker."""
    acct_path, act_path, data_path, fsl_path = synthetic_csvs
    save_path = str(tmp_path / "out.png")

    plot_backtest(acct_path, act_path, data_path=data_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    # Expect 2 Line2D objects (one per ticker: AAPL, MSFT)
    lines = [c for c in ax3.get_children() if isinstance(c, plt.Line2D)]
    # Filter out grid lines (they have no label or label starting with '_')
    named_lines = [line for line in lines if line.get_label() and not line.get_label().startswith("_")]
    assert len(named_lines) >= 2, f"Expected ≥2 ticker lines, got {len(named_lines)}"
    plt.close("all")


def test_plot_backtest_with_data_path_scatter_buy_sell(synthetic_csvs, tmp_path):
    """When data_path is provided, buy/sell scatter markers should appear on the third axis."""
    acct_path, act_path, data_path, fsl_path = synthetic_csvs
    save_path = str(tmp_path / "out.png")

    plot_backtest(acct_path, act_path, data_path=data_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    from matplotlib.collections import PathCollection
    scatters = [c for c in ax3.get_children() if isinstance(c, PathCollection)]
    # At least some buy and sell markers should exist
    assert len(scatters) >= 1, "Expected at least one scatter (buy or sell) on the third axis"
    plt.close("all")


def test_plot_backtest_without_data_path_falls_back(synthetic_csvs, tmp_path):
    """Without data_path, the third axis should still plot asset_value (backward compat)."""
    acct_path, act_path, data_path, fsl_path = synthetic_csvs
    save_path = str(tmp_path / "out.png")

    # Call without data_path – should NOT raise
    plot_backtest(acct_path, act_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    assert "Market Exposure" in ax3.get_title()
    plt.close("all")


def test_plot_backtest_third_axis_title_with_data_path(synthetic_csvs, tmp_path):
    """When data_path is given, third axis title should reference ticker prices."""
    acct_path, act_path, data_path, fsl_path = synthetic_csvs
    save_path = str(tmp_path / "out.png")

    plot_backtest(acct_path, act_path, data_path=data_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    assert "Ticker Prices" in ax3.get_title(), f"Got title: '{ax3.get_title()}'"
    plt.close("all")


def test_plot_stoploss_markers_distinct(tmp_path):
    """When stoploss_mask column is present, stop-loss sells should get a distinct
    'Stop Loss' labelled scatter marker, separate from normal 'Sell' markers."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="US/Eastern")

    # --- account_history CSV ---
    acct_rows = [{"timestamp": d, "cash": 900_000, "asset_value": 100_000,
                  "total_assets": 1_000_000, "reward": 0.0} for d in dates]
    df_acct = pd.DataFrame(acct_rows)
    acct_path = str(tmp_path / "account_history.csv")
    df_acct.to_csv(acct_path, index=False)

    # --- action_history CSV with stoploss_mask ---
    act_rows = []
    for i, d in enumerate(dates):
        if i == 0:
            trans, sl = "[10.0 5.0]", "[0.0 0.0]"    # buy both
        elif i == 2:
            trans, sl = "[-5.0 0.0]", "[1.0 0.0]"     # stop-loss sell AAPL
        elif i == 3:
            trans, sl = "[-3.0 0.0]", "[0.0 0.0]"     # normal sell AAPL
        else:
            trans, sl = "[0.0 0.0]", "[0.0 0.0]"       # hold
        act_rows.append({"timestamp": d, "actions": "[0.0 0.0]",
                         "transactions": trans, "stoploss_mask": sl})
    df_act = pd.DataFrame(act_rows)
    act_path = str(tmp_path / "action_history.csv")
    df_act.to_csv(act_path, index=False)

    # --- original data CSV ---
    data_rows = []
    for d in dates:
        for tic, base in [("AAPL", 150.0), ("MSFT", 380.0)]:
            data_rows.append({"date": d, "tic": tic, "open": base, "high": base + 3,
                              "low": base - 3, "close": base, "volume": 1_000_000})
    df_data = pd.DataFrame(data_rows)
    data_path = str(tmp_path / "data.csv")
    df_data.to_csv(data_path, index=False)

    save_path = str(tmp_path / "out.png")
    plot_backtest(acct_path, act_path, data_path=data_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    from matplotlib.collections import PathCollection
    scatters = [c for c in ax3.get_children() if isinstance(c, PathCollection)]
    labels = [s.get_label() for s in scatters]
    assert "Stop Loss" in labels, (
        f"Expected a scatter with label 'Stop Loss', got labels: {labels}"
    )
    plt.close("all")


def test_plot_backward_compat_no_stoploss_column(tmp_path):
    """When stoploss_mask column is absent (old CSVs), plot should still work
    with no errors and no 'Stop Loss' markers."""
    dates = pd.date_range("2024-01-02", periods=3, freq="B", tz="US/Eastern")

    acct_rows = [{"timestamp": d, "cash": 900_000, "asset_value": 100_000,
                  "total_assets": 1_000_000, "reward": 0.0} for d in dates]
    df_acct = pd.DataFrame(acct_rows)
    acct_path = str(tmp_path / "account_history.csv")
    df_acct.to_csv(acct_path, index=False)

    act_rows = [{"timestamp": d, "actions": "[0.0 0.0]",
                 "transactions": f"[{-5.0 if i == 1 else 10.0} 0.0]"}
                for i, d in enumerate(dates)]
    df_act = pd.DataFrame(act_rows)
    act_path = str(tmp_path / "action_history.csv")
    df_act.to_csv(act_path, index=False)

    data_rows = []
    for d in dates:
        for tic, base in [("AAPL", 150.0), ("MSFT", 380.0)]:
            data_rows.append({"date": d, "tic": tic, "open": base, "high": base + 3,
                              "low": base - 3, "close": base, "volume": 1_000_000})
    df_data = pd.DataFrame(data_rows)
    data_path = str(tmp_path / "data.csv")
    df_data.to_csv(data_path, index=False)

    save_path = str(tmp_path / "out.png")
    # Should not raise
    plot_backtest(acct_path, act_path, data_path=data_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    from matplotlib.collections import PathCollection
    scatters = [c for c in ax3.get_children() if isinstance(c, PathCollection)]
    labels = [s.get_label() for s in scatters]
    assert "Stop Loss" not in labels, (
        "Old CSVs without stoploss_mask should not produce 'Stop Loss' markers"
    )
    plt.close("all")


def test_plot_backtest_with_fixed_sl_path(synthetic_csvs, tmp_path):
    """When fixed_sl_path is provided, it should overlay lines on ax0 and ax1."""
    acct_path, act_path, data_path, fsl_path = synthetic_csvs
    save_path = str(tmp_path / "out.png")

    # Should not raise
    plot_backtest(acct_path, act_path, fixed_sl_path=fsl_path, save_path=save_path)

    fig = plt.gcf()
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]

    # Check that ax1 has the PPO + Fixed SL line (with percentage)
    lines1 = [c for c in ax1.get_children() if isinstance(c, plt.Line2D)]
    labels1 = [line.get_label() for line in lines1]
    assert "PPO + Fixed SL (95%)" in labels1

    # Check that ax2 has the PPO + Fixed SL Drawdown line (with percentage)
    lines2 = [c for c in ax2.get_children() if isinstance(c, plt.Line2D)]
    labels2 = [line.get_label() for line in lines2]
    assert "PPO + Fixed SL (95%)" in labels2

    plt.close("all")


def test_plot_backtest_legend_order(synthetic_csvs, tmp_path):
    """Legend entries should appear in the specified order:
    PPO Agent, PPO + Fixed SL (%), Buy & Hold Baseline, Initial ($)."""
    acct_path, act_path, data_path, fsl_path = synthetic_csvs

    # Create a baseline CSV (same shape as account_history but with 'date' column)
    df_acct = pd.read_csv(acct_path, parse_dates=["timestamp"])
    df_bl = df_acct.rename(columns={"timestamp": "date"}).copy()
    df_bl["total_assets"] = df_bl["total_assets"] * 0.98
    bl_path = str(tmp_path / "baseline.csv")
    df_bl.to_csv(bl_path, index=False)

    save_path = str(tmp_path / "out.png")
    plot_backtest(
        acct_path, act_path,
        baseline_path=bl_path,
        fixed_sl_path=fsl_path,
        save_path=save_path,
    )

    fig = plt.gcf()
    ax1 = fig.axes[0]
    handles, labels = ax1.get_legend_handles_labels()

    assert labels[0] == "PPO Agent"
    assert labels[1] == "PPO + Fixed SL (95%)"
    assert labels[2] == "Buy & Hold Baseline"
    assert labels[3].startswith("Initial (")

    plt.close("all")
