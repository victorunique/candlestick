"""Tests for src/wilcoxon_test.py — Wilcoxon signed-rank tests for strategy comparison.

The test fixtures simulate the real data structure where multiple
reward_weight_drawdown settings exist per (ticker, window) pair.
The implementation must aggregate by (ticker, window) before running
the Wilcoxon test to avoid pseudoreplication.
"""

import pandas as pd
import pytest

from src import wilcoxon_test


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a DataFrame with multiple weights per (ticker, window).

    Simulates the real structure: 3 weights × 3 tickers × 4 windows = 36 rows.
    PPO clearly dominates FSL and B&H after aggregation.
    BH values are identical within each (ticker, window) — matching real data.
    FSL values vary by weight — matching real data (different trained models).
    """
    rows = []
    tickers = ["AAPL", "MSFT", "NVDA"]
    windows = [
        ("2026-02-24", "2026-02-25"),
        ("2026-02-26", "2026-02-27"),
        ("2026-03-03", "2026-03-04"),
        ("2026-03-05", "2026-03-06"),
    ]
    weights = [0.10, 0.20, 0.30]
    for t_start, t_end in windows:
        for ticker in tickers:
            for i, w in enumerate(weights):
                rows.append({
                    "combo_id": 1,
                    "test_start": t_start, "test_end": t_end,
                    "tickers": ticker,
                    "reward_weight_drawdown": w,
                    # PPO varies by weight (different trained models)
                    "ppo_return": 0.05 + 0.001 * i,
                    "ppo_max_dd": -0.01 - 0.001 * i,
                    "ppo_sortino": 0.08 + 0.002 * i,
                    # FSL varies by weight (same model, fixed SL)
                    "fsl_return": 0.02 + 0.001 * i,
                    "fsl_max_dd": -0.03 - 0.001 * i,
                    "fsl_sortino": 0.03 + 0.001 * i,
                    # BH is identical across weights (no model dependency)
                    "bh_return": 0.01,
                    "bh_max_dd": -0.04,
                    "bh_sortino": 0.02,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def identical_df():
    """DataFrame where PPO and FSL are identical after aggregation."""
    rows = []
    tickers = ["AAPL", "MSFT"]
    windows = [(f"2026-01-{d:02d}", f"2026-01-{d+1:02d}") for d in range(1, 11)]
    weights = [0.10, 0.20]
    for t_start, t_end in windows:
        for ticker in tickers:
            for i, w in enumerate(weights):
                val = hash((t_start, ticker)) % 100 / 1000  # deterministic variation
                rows.append({
                    "combo_id": 1,
                    "test_start": t_start, "test_end": t_end,
                    "tickers": ticker,
                    "reward_weight_drawdown": w,
                    "ppo_return": val + 0.001 * i,
                    "ppo_max_dd": -val - 0.001 * i,
                    "ppo_sortino": val + 0.002 * i,
                    "fsl_return": val + 0.001 * i,      # identical to PPO
                    "fsl_max_dd": -val - 0.001 * i,     # identical to PPO
                    "fsl_sortino": val + 0.002 * i,     # identical to PPO
                    "bh_return": val,
                    "bh_max_dd": -val,
                    "bh_sortino": val,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    """Write sample_df to a CSV file."""
    path = str(tmp_path / "test_results.csv")
    sample_df.to_csv(path, index=False)
    return path


# ── Tests: load_data ─────────────────────────────────────────────────────

class TestLoadData:
    def test_loads_valid_csv(self, sample_csv):
        df = wilcoxon_test.load_data(sample_csv)
        assert len(df) == 36  # 3 weights × 3 tickers × 4 windows

    def test_raises_on_missing_columns(self, tmp_path):
        bad_csv = str(tmp_path / "bad.csv")
        pd.DataFrame({"col_a": [1]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            wilcoxon_test.load_data(bad_csv)


# ── Tests: aggregate_by_window ───────────────────────────────────────────

class TestAggregateByWindow:
    def test_output_has_one_row_per_ticker_window(self, sample_df):
        """3 tickers × 4 windows = 12 rows after aggregation."""
        agg = wilcoxon_test.aggregate_by_window(sample_df)
        assert len(agg) == 12

    def test_aggregated_columns_present(self, sample_df):
        agg = wilcoxon_test.aggregate_by_window(sample_df)
        for prefix in ("ppo", "fsl", "bh"):
            for suffix in ("return", "max_dd", "sortino"):
                assert f"{prefix}_{suffix}" in agg.columns

    def test_bh_values_unchanged(self, sample_df):
        """BH values are identical across weights, so mean == original."""
        agg = wilcoxon_test.aggregate_by_window(sample_df)
        assert (agg["bh_return"] == 0.01).all()
        assert (agg["bh_max_dd"] == -0.04).all()

    def test_ppo_values_are_averaged(self, sample_df):
        """PPO values should be the mean across weights."""
        agg = wilcoxon_test.aggregate_by_window(sample_df)
        # ppo_return values: 0.05, 0.051, 0.052 → mean = 0.051
        expected_mean = (0.05 + 0.051 + 0.052) / 3
        assert abs(agg["ppo_return"].iloc[0] - expected_mean) < 1e-9


# ── Tests: run_wilcoxon ──────────────────────────────────────────────────

class TestRunWilcoxon:
    def test_returns_dict_with_expected_keys(self, sample_df):
        agg = wilcoxon_test.aggregate_by_window(sample_df)
        result = wilcoxon_test.run_wilcoxon(
            agg["ppo_return"], agg["fsl_return"],
        )
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result

    def test_significant_when_ppo_dominates(self, sample_df):
        agg = wilcoxon_test.aggregate_by_window(sample_df)
        result = wilcoxon_test.run_wilcoxon(
            agg["ppo_return"], agg["fsl_return"],
        )
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_not_significant_when_identical(self, identical_df):
        agg = wilcoxon_test.aggregate_by_window(identical_df)
        result = wilcoxon_test.run_wilcoxon(
            agg["ppo_return"], agg["fsl_return"],
        )
        # p-value should be 1.0 when all differences are zero
        assert result["significant"] is False


# ── Tests: build_results_table ───────────────────────────────────────────

class TestBuildResultsTable:
    def test_returns_dataframe_with_six_rows(self, sample_df):
        """2 comparisons × 3 metrics = 6 rows."""
        table = wilcoxon_test.build_results_table(sample_df)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 6

    def test_has_required_columns(self, sample_df):
        table = wilcoxon_test.build_results_table(sample_df)
        expected_cols = {"Comparison", "Metric", "Test", "p-value", "Significant"}
        assert expected_cols == set(table.columns)

    def test_all_comparisons_present(self, sample_df):
        table = wilcoxon_test.build_results_table(sample_df)
        comparisons = set(table["Comparison"])
        assert "PPO vs Fixed SL" in comparisons
        assert "PPO vs Buy&Hold" in comparisons

    def test_all_metrics_present(self, sample_df):
        table = wilcoxon_test.build_results_table(sample_df)
        metrics = set(table["Metric"])
        assert metrics == {"Return", "Max Drawdown", "Sortino"}

    def test_test_column_is_wilcoxon(self, sample_df):
        table = wilcoxon_test.build_results_table(sample_df)
        assert all(table["Test"] == "Wilcoxon")


# ── Tests: CLI entry point ───────────────────────────────────────────────

class TestMain:
    def test_main_runs_without_error(self, sample_csv, capsys):
        wilcoxon_test.main(["--input", sample_csv])
        captured = capsys.readouterr()
        assert "Wilcoxon Signed-Rank Test Results" in captured.out
        assert "PPO vs Fixed SL" in captured.out
        assert "PPO vs Buy&Hold" in captured.out

    def test_main_reports_aggregated_sample_size(self, sample_csv, capsys):
        """Should report N=12 (3 tickers × 4 windows), not N=36."""
        wilcoxon_test.main(["--input", sample_csv])
        captured = capsys.readouterr()
        assert "Number of paired observations: 12" in captured.out
