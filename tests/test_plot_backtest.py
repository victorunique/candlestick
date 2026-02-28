"""Tests for plot_backtest module – TDD Red phase for ticker-price third graph."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.plot_backtest import plot_backtest


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_csvs(tmp_path):
    """Create minimal synthetic CSVs (account, action, data) for testing.
    
    Uses timezone-aware timestamps (like real backtest output) to exercise
    the tz normalisation logic.
    """
    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="US/Eastern")
    tickers = ["AAPL", "MSFT"]  # sorted order

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

    return acct_path, act_path, data_path


# ── Tests ────────────────────────────────────────────────────────────────

def test_plot_backtest_with_data_path_creates_ticker_lines(synthetic_csvs, tmp_path):
    """When data_path is provided, the third axis should have one line per ticker."""
    acct_path, act_path, data_path = synthetic_csvs
    save_path = str(tmp_path / "out.png")

    plot_backtest(acct_path, act_path, data_path=data_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    # Expect 2 Line2D objects (one per ticker: AAPL, MSFT)
    lines = [c for c in ax3.get_children() if isinstance(c, plt.Line2D)]
    # Filter out grid lines (they have no label or label starting with '_')
    named_lines = [l for l in lines if l.get_label() and not l.get_label().startswith("_")]
    assert len(named_lines) >= 2, f"Expected ≥2 ticker lines, got {len(named_lines)}"
    plt.close("all")


def test_plot_backtest_with_data_path_scatter_buy_sell(synthetic_csvs, tmp_path):
    """When data_path is provided, buy/sell scatter markers should appear on the third axis."""
    acct_path, act_path, data_path = synthetic_csvs
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
    acct_path, act_path, data_path = synthetic_csvs
    save_path = str(tmp_path / "out.png")

    # Call without data_path – should NOT raise
    plot_backtest(acct_path, act_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    assert "Market Exposure" in ax3.get_title()
    plt.close("all")


def test_plot_backtest_third_axis_title_with_data_path(synthetic_csvs, tmp_path):
    """When data_path is given, third axis title should reference ticker prices."""
    acct_path, act_path, data_path = synthetic_csvs
    save_path = str(tmp_path / "out.png")

    plot_backtest(acct_path, act_path, data_path=data_path, save_path=save_path)

    fig = plt.gcf()
    ax3 = fig.axes[2]
    assert "Ticker Prices" in ax3.get_title(), f"Got title: '{ax3.get_title()}'"
    plt.close("all")
