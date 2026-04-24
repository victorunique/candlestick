"""Tests for src/analyse_pareto.py."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from src import analyse_pareto


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal CSV with all required columns for testing."""
    rows = []
    weights = [0.05, 0.10, 0.20, 0.30]
    tickers = ["AAPL", "MSFT"]
    for w in weights:
        for ticker in tickers:
            rows.append({
                "combo_id": 1,
                "train_start": "2026-01-01", "train_end": "2026-01-15",
                "test_start": "2026-01-15", "test_end": "2026-01-16",
                "tickers": ticker,
                "total_timesteps": 100000,
                "reward_weight_pnl": 1.0,
                "reward_weight_drawdown": w,
                "cash_penalty_proportion": 0.0,
                "upside_pnl_multiplier": 1.25,
                "stoploss_min": 0.75,
                "stoploss_max": 1.0,
                "n_steps": 2048,
                "ent_coef": 0.01,
                "learning_rate": 0.0001,
                "gamma": 0.99,
                "episode_length": 500,
                "ppo_return": 0.01 * (1 / w),     # higher w → lower return
                "ppo_max_dd": -0.01 * w,           # higher w → less DD
                "ppo_sharpe": 0.03 * (1 / w),
                "ppo_sortino": 0.05 * (1 / w),
                "fsl_return": 0.005 * (1 / w),
                "fsl_max_dd": -0.015 * w,
                "fsl_sharpe": 0.02 * (1 / w),
                "fsl_sortino": 0.04 * (1 / w),
                "bh_return": 0.008,
                "bh_max_dd": -0.012,
                "bh_sharpe": 0.025,
                "bh_sortino": 0.04,
            })
    df = pd.DataFrame(rows)
    path = str(tmp_path / "test_results.csv")
    df.to_csv(path, index=False)
    return path


# ── Tests ─────────────────────────────────────────────────────────────────

class TestLoadData:
    def test_loads_valid_csv(self, sample_csv):
        df = analyse_pareto.load_data(sample_csv)
        assert len(df) == 8  # 4 weights × 2 tickers

    def test_validates_required_columns(self, tmp_path):
        bad_csv = str(tmp_path / "bad.csv")
        pd.DataFrame({"col_a": [1]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            analyse_pareto.load_data(bad_csv)

    def test_requires_sharpe_sortino_columns(self, tmp_path):
        """New columns ppo_sharpe, ppo_sortino etc. must be required."""
        bad_csv = str(tmp_path / "missing_sharpe.csv")
        pd.DataFrame({
            "reward_weight_drawdown": [0.1],
            "ppo_return": [0.01],
            "ppo_max_dd": [-0.01],
            # Missing: ppo_sharpe, ppo_sortino, fsl_*, bh_*
        }).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            analyse_pareto.load_data(bad_csv)


class TestAggregateByWeight:
    def test_aggregation_includes_sharpe_sortino(self, sample_csv):
        df = analyse_pareto.load_data(sample_csv)
        agg = analyse_pareto.aggregate_by_weight(df)
        expected_cols = {
            "avg_return", "avg_max_dd", "worst_max_dd",
            "median_return", "count",
            "avg_ppo_sharpe", "avg_ppo_sortino",
            "avg_fsl_sharpe", "avg_fsl_sortino",
            "avg_bh_sharpe", "avg_bh_sortino",
        }
        assert expected_cols.issubset(set(agg.columns))

    def test_aggregation_row_count(self, sample_csv):
        df = analyse_pareto.load_data(sample_csv)
        agg = analyse_pareto.aggregate_by_weight(df)
        assert len(agg) == 4  # 4 unique weights

    def test_aggregation_values(self, sample_csv):
        """Check that aggregation averages are correct for a known weight."""
        df = analyse_pareto.load_data(sample_csv)
        agg = analyse_pareto.aggregate_by_weight(df)
        row_010 = agg[agg["reward_weight_drawdown"] == 0.10].iloc[0]
        # For w=0.10: ppo_return = 0.01 * (1/0.1) = 0.1, same for both tickers
        assert abs(row_010["avg_return"] - 0.1) < 1e-9
        # ppo_sharpe = 0.03 * (1/0.1) = 0.3
        assert abs(row_010["avg_ppo_sharpe"] - 0.3) < 1e-9
        # ppo_sortino = 0.05 * (1/0.1) = 0.5
        assert abs(row_010["avg_ppo_sortino"] - 0.5) < 1e-9


class TestParetoFront:
    def test_pareto_front_basic(self):
        # Point 0: dominated by point 1 (lower in both)
        # Point 1: Pareto-optimal (highest return, moderate DD)
        # Point 2: Pareto-optimal (moderate return, least DD)
        returns = np.array([0.01, 0.05, 0.03])
        dds = np.array([-0.05, -0.03, -0.01])
        pf = analyse_pareto.pareto_front(returns, dds)
        assert set(pf) == {1, 2}

    def test_single_point_is_pareto(self):
        pf = analyse_pareto.pareto_front(np.array([0.1]), np.array([-0.01]))
        assert list(pf) == [0]


class TestPlotPareto:
    def test_generates_output_files(self, sample_csv, tmp_path):
        """plot_pareto should generate the main plot and the metrics comparison."""
        df = analyse_pareto.load_data(sample_csv)
        agg = analyse_pareto.aggregate_by_weight(df)
        output = str(tmp_path / "test_pareto.png")
        analyse_pareto.plot_pareto(agg, output)
        assert os.path.exists(output)

    def test_generates_sharpe_comparison(self, sample_csv, tmp_path):
        """plot_sharpe_comparison should generate the Sharpe chart."""
        df = analyse_pareto.load_data(sample_csv)
        agg = analyse_pareto.aggregate_by_weight(df)
        output = str(tmp_path / "test_sharpe.png")
        analyse_pareto.plot_sharpe_comparison(agg, output)
        assert os.path.exists(output)

    def test_generates_sortino_comparison(self, sample_csv, tmp_path):
        """plot_sortino_comparison should generate the Sortino chart."""
        df = analyse_pareto.load_data(sample_csv)
        agg = analyse_pareto.aggregate_by_weight(df)
        output = str(tmp_path / "test_sortino.png")
        analyse_pareto.plot_sortino_comparison(agg, output)
        assert os.path.exists(output)

