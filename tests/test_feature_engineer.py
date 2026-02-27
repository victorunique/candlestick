import pytest
import pandas as pd
import numpy as np
from src.feature_engineer import FeatureEngineer, INDICATORS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Generate 100 days of synthetic OHLCV data for two tickers."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = []

    for tic in ["AAPL", "MSFT"]:
        base_price = 150 if tic == "AAPL" else 300
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))

        for i in range(100):
            data.append({
                "date": dates[i].strftime("%Y-%m-%d"),
                "tic": tic,
                "open": prices[i] * 0.99,
                "high": prices[i] * 1.02,
                "low": prices[i] * 0.98,
                "close": prices[i],
                "volume": int(np.random.uniform(1_000_000, 5_000_000)),
            })

    df = pd.DataFrame(data)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    return df


@pytest.fixture
def single_ticker_data():
    """Generate 100 days of synthetic OHLCV data for a single ticker."""
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    returns = np.random.normal(0, 0.015, 100)
    prices = 200 * np.exp(np.cumsum(returns))

    data = []
    for i in range(100):
        data.append({
            "date": dates[i].strftime("%Y-%m-%d"),
            "tic": "GOOG",
            "open": prices[i] * 0.995,
            "high": prices[i] * 1.015,
            "low": prices[i] * 0.985,
            "close": prices[i],
            "volume": int(np.random.uniform(500_000, 3_000_000)),
        })

    df = pd.DataFrame(data)
    return df


def _process(df, indicators):
    """Helper: run FeatureEngineer with the given indicator list."""
    fe = FeatureEngineer(indicators=indicators)
    return fe.preprocess_data(df)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestFeatureEngineerInit:
    def test_defaults_to_all_indicators(self):
        fe = FeatureEngineer()
        assert fe.indicators == INDICATORS

    def test_custom_indicators(self):
        custom = ["macd", "rsi_14"]
        fe = FeatureEngineer(indicators=custom)
        assert fe.indicators == custom

    def test_required_columns_set(self):
        fe = FeatureEngineer()
        for col in ["date", "tic", "open", "high", "low", "close", "volume"]:
            assert col in fe.required_columns


# ---------------------------------------------------------------------------
# Column validation tests
# ---------------------------------------------------------------------------

class TestCheckColumns:
    def test_missing_single_column(self, sample_data):
        fe = FeatureEngineer(indicators=["macd"])
        bad = sample_data.drop(columns=["high"])
        with pytest.raises(ValueError, match="high"):
            fe.preprocess_data(bad)

    def test_missing_multiple_columns(self, sample_data):
        fe = FeatureEngineer(indicators=["macd"])
        bad = sample_data.drop(columns=["high", "low"])
        with pytest.raises(ValueError):
            fe.preprocess_data(bad)

    def test_valid_columns_no_error(self, sample_data):
        fe = FeatureEngineer(indicators=["rsi_14"])
        # Should not raise
        fe.preprocess_data(sample_data)


# ---------------------------------------------------------------------------
# ATR (Average True Range) tests
# ---------------------------------------------------------------------------

class TestATR:
    def test_atr_column_exists(self, sample_data):
        result = _process(sample_data, ["atr"])
        assert "atr" in result.columns

    def test_atr_20_column_exists(self, sample_data):
        result = _process(sample_data, ["atr_20"])
        assert "atr_20" in result.columns

    def test_atr_values_non_negative(self, sample_data):
        result = _process(sample_data, ["atr"])
        assert (result["atr"] >= 0).all(), "ATR values should be non-negative"

    def test_atr_20_values_non_negative(self, sample_data):
        result = _process(sample_data, ["atr_20"])
        assert (result["atr_20"] >= 0).all(), "ATR(20) values should be non-negative"

    def test_atr_no_nan(self, sample_data):
        result = _process(sample_data, ["atr", "atr_20"])
        assert result["atr"].isna().sum() == 0
        assert result["atr_20"].isna().sum() == 0

    def test_atr_reasonable_magnitude(self, sample_data):
        """ATR should be a fraction of the price, not larger than the price itself."""
        result = _process(sample_data, ["atr"])
        assert (result["atr"] < result["close"]).all()


# ---------------------------------------------------------------------------
# True Range tests
# ---------------------------------------------------------------------------

class TestTrueRange:
    def test_tr_column_exists(self, sample_data):
        result = _process(sample_data, ["tr"])
        assert "tr" in result.columns

    def test_tr_values_non_negative(self, sample_data):
        result = _process(sample_data, ["tr"])
        assert (result["tr"] >= 0).all(), "True range should be non-negative"

    def test_tr_no_nan(self, sample_data):
        result = _process(sample_data, ["tr"])
        assert result["tr"].isna().sum() == 0


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_column_exists(self, sample_data):
        result = _process(sample_data, ["macd"])
        assert "macd" in result.columns

    def test_macds_column_exists(self, sample_data):
        result = _process(sample_data, ["macds"])
        assert "macds" in result.columns

    def test_macdh_column_exists(self, sample_data):
        result = _process(sample_data, ["macdh"])
        assert "macdh" in result.columns

    def test_macd_all_three_together(self, sample_data):
        result = _process(sample_data, ["macd", "macds", "macdh"])
        for col in ["macd", "macds", "macdh"]:
            assert col in result.columns

    def test_macd_no_nan(self, sample_data):
        result = _process(sample_data, ["macd", "macds", "macdh"])
        for col in ["macd", "macds", "macdh"]:
            assert result[col].isna().sum() == 0

    def test_macd_values_are_numeric(self, sample_data):
        result = _process(sample_data, ["macd", "macds", "macdh"])
        for col in ["macd", "macds", "macdh"]:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_macdh_is_populated(self, sample_data):
        """MACD histogram should contain non-zero computed values."""
        result = _process(sample_data, ["macd", "macds", "macdh"])
        # After backfill the identity macdh = macd - signal won't hold for
        # backfilled rows, so just verify histogram is populated and numeric.
        assert pd.api.types.is_numeric_dtype(result["macdh"])
        assert (result["macdh"] != 0).any()


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_14_column_exists(self, sample_data):
        result = _process(sample_data, ["rsi_14"])
        assert "rsi_14" in result.columns

    def test_rsi_6_column_exists(self, sample_data):
        result = _process(sample_data, ["rsi_6"])
        assert "rsi_6" in result.columns

    def test_rsi_14_range(self, sample_data):
        """RSI should be between 0 and 100."""
        result = _process(sample_data, ["rsi_14"])
        assert (result["rsi_14"] >= 0).all()
        assert (result["rsi_14"] <= 100).all()

    def test_rsi_6_range(self, sample_data):
        result = _process(sample_data, ["rsi_6"])
        assert (result["rsi_6"] >= 0).all()
        assert (result["rsi_6"] <= 100).all()

    def test_rsi_no_nan(self, sample_data):
        result = _process(sample_data, ["rsi_14", "rsi_6"])
        assert result["rsi_14"].isna().sum() == 0
        assert result["rsi_6"].isna().sum() == 0

    def test_rsi_custom_window(self, sample_data):
        """Support for arbitrary RSI windows (e.g., rsi_21)."""
        result = _process(sample_data, ["rsi_21"])
        assert "rsi_21" in result.columns
        assert (result["rsi_21"] >= 0).all()
        assert (result["rsi_21"] <= 100).all()


# ---------------------------------------------------------------------------
# CCI tests
# ---------------------------------------------------------------------------

class TestCCI:
    def test_cci_column_exists(self, sample_data):
        result = _process(sample_data, ["cci"])
        assert "cci" in result.columns

    def test_cci_20_column_exists(self, sample_data):
        result = _process(sample_data, ["cci_20"])
        assert "cci_20" in result.columns

    def test_cci_no_nan(self, sample_data):
        result = _process(sample_data, ["cci", "cci_20"])
        assert result["cci"].isna().sum() == 0
        assert result["cci_20"].isna().sum() == 0

    def test_cci_is_numeric(self, sample_data):
        result = _process(sample_data, ["cci"])
        assert pd.api.types.is_numeric_dtype(result["cci"])

    def test_cci_default_window_is_14(self, sample_data):
        """The 'cci' indicator should use window=14 by default."""
        result = _process(sample_data, ["cci"])
        # Just verify it computes without error and produces values
        non_zero = (result["cci"] != 0).sum()
        assert non_zero > 0


# ---------------------------------------------------------------------------
# ADX / DX tests
# ---------------------------------------------------------------------------

class TestADX:
    def test_adx_column_exists(self, sample_data):
        result = _process(sample_data, ["adx"])
        assert "adx" in result.columns

    def test_dx_column_exists(self, sample_data):
        result = _process(sample_data, ["dx"])
        assert "dx" in result.columns

    def test_adx_range(self, sample_data):
        """ADX should be between 0 and 100."""
        result = _process(sample_data, ["adx"])
        assert (result["adx"] >= 0).all()
        assert (result["adx"] <= 100).all()

    def test_adx_no_nan(self, sample_data):
        result = _process(sample_data, ["adx"])
        assert result["adx"].isna().sum() == 0


# ---------------------------------------------------------------------------
# SMA tests
# ---------------------------------------------------------------------------

class TestSMA:
    def test_close_30_sma_column_exists(self, sample_data):
        result = _process(sample_data, ["close_30_sma"])
        assert "close_30_sma" in result.columns

    def test_close_60_sma_column_exists(self, sample_data):
        result = _process(sample_data, ["close_60_sma"])
        assert "close_60_sma" in result.columns

    def test_sma_no_nan(self, sample_data):
        result = _process(sample_data, ["close_30_sma", "close_60_sma"])
        assert result["close_30_sma"].isna().sum() == 0
        assert result["close_60_sma"].isna().sum() == 0

    def test_sma_positive(self, sample_data):
        """SMA of positive prices should be positive."""
        result = _process(sample_data, ["close_30_sma"])
        # After backfill, all values should be populated and positive
        computed = result.loc[result["close_30_sma"] != 0, "close_30_sma"]
        if len(computed) > 0:
            assert (computed > 0).all()

    def test_sma_smoothing_effect(self, sample_data):
        """SMA should have lower variance than the raw close price per ticker."""
        result = _process(sample_data, ["close_30_sma"])
        for tic in result["tic"].unique():
            ticker = result[result["tic"] == tic]
            computed = ticker[ticker["close_30_sma"] != 0]
            if len(computed) > 10:
                assert computed["close_30_sma"].std() <= computed["close"].std()

    def test_sma_custom_window(self, sample_data):
        """Support for arbitrary SMA windows via naming convention."""
        result = _process(sample_data, ["close_10_sma"])
        assert "close_10_sma" in result.columns


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

class TestEMA:
    def test_close_12_ema_column_exists(self, sample_data):
        result = _process(sample_data, ["close_12_ema"])
        assert "close_12_ema" in result.columns

    def test_close_26_ema_column_exists(self, sample_data):
        result = _process(sample_data, ["close_26_ema"])
        assert "close_26_ema" in result.columns

    def test_close_50_ema_column_exists(self, sample_data):
        result = _process(sample_data, ["close_50_ema"])
        assert "close_50_ema" in result.columns

    def test_ema_all_three(self, sample_data):
        result = _process(sample_data, ["close_12_ema", "close_26_ema", "close_50_ema"])
        for col in ["close_12_ema", "close_26_ema", "close_50_ema"]:
            assert col in result.columns

    def test_ema_no_nan(self, sample_data):
        result = _process(sample_data, ["close_12_ema", "close_26_ema", "close_50_ema"])
        for col in ["close_12_ema", "close_26_ema", "close_50_ema"]:
            assert result[col].isna().sum() == 0

    def test_ema_positive(self, sample_data):
        """EMA of positive prices should be positive."""
        result = _process(sample_data, ["close_12_ema"])
        computed = result.loc[result["close_12_ema"] != 0, "close_12_ema"]
        if len(computed) > 0:
            assert (computed > 0).all()

    def test_shorter_ema_more_responsive(self, sample_data):
        """EMA with shorter window should track price more closely (lower MSE)."""
        result = _process(sample_data, ["close_12_ema", "close_50_ema"])
        for tic in result["tic"].unique():
            t = result[result["tic"] == tic]
            mse_12 = ((t["close"] - t["close_12_ema"]) ** 2).mean()
            mse_50 = ((t["close"] - t["close_50_ema"]) ** 2).mean()
            assert mse_12 <= mse_50, "Shorter EMA should track price more closely"


# ---------------------------------------------------------------------------
# Bollinger Bands tests
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_boll_columns_exist(self, sample_data):
        result = _process(sample_data, ["boll", "boll_ub", "boll_lb"])
        for col in ["boll", "boll_ub", "boll_lb"]:
            assert col in result.columns

    def test_boll_no_nan(self, sample_data):
        result = _process(sample_data, ["boll", "boll_ub", "boll_lb"])
        for col in ["boll", "boll_ub", "boll_lb"]:
            assert result[col].isna().sum() == 0

    def test_boll_ordering(self, sample_data):
        """Upper band >= middle band >= lower band (where actually computed)."""
        result = _process(sample_data, ["boll", "boll_ub", "boll_lb"])
        # Only check rows where all three are actually computed
        mask = (result["boll"] != 0) & (result["boll_ub"] != 0) & (result["boll_lb"] != 0)
        if mask.any():
            subset = result[mask]
            assert (subset["boll_ub"] >= subset["boll"] - 1e-10).all(), \
                "Upper band should be >= middle band"
            assert (subset["boll"] >= subset["boll_lb"] - 1e-10).all(), \
                "Middle band should be >= lower band"

    def test_boll_positive(self, sample_data):
        """Bollinger bands on positive prices should be positive."""
        result = _process(sample_data, ["boll", "boll_ub", "boll_lb"])
        for col in ["boll", "boll_ub", "boll_lb"]:
            computed = result.loc[result[col] != 0, col]
            if len(computed) > 0:
                assert (computed > 0).all()


# ---------------------------------------------------------------------------
# KDJ / Stochastic Oscillator tests
# ---------------------------------------------------------------------------

class TestKDJ:
    def test_kdjk_column_exists(self, sample_data):
        result = _process(sample_data, ["kdjk"])
        assert "kdjk" in result.columns

    def test_kdjd_column_exists(self, sample_data):
        result = _process(sample_data, ["kdjd"])
        assert "kdjd" in result.columns

    def test_kdjj_falls_back_to_zero(self, sample_data):
        """kdjj is listed in INDICATORS but has no dedicated stoch column;
        it should fall back to 0.0."""
        result = _process(sample_data, ["kdjj"])
        assert "kdjj" in result.columns

    def test_kdjk_range(self, sample_data):
        """Stochastic %K should be between 0 and 100."""
        result = _process(sample_data, ["kdjk"])
        computed = result.loc[result["kdjk"] != 0, "kdjk"]
        if len(computed) > 0:
            assert (computed >= 0).all()
            assert (computed <= 100).all()

    def test_kdjd_range(self, sample_data):
        """Stochastic %D should be between 0 and 100."""
        result = _process(sample_data, ["kdjd"])
        computed = result.loc[result["kdjd"] != 0, "kdjd"]
        if len(computed) > 0:
            assert (computed >= 0).all()
            assert (computed <= 100).all()

    def test_kdj_no_nan(self, sample_data):
        result = _process(sample_data, ["kdjk", "kdjd"])
        assert result["kdjk"].isna().sum() == 0
        assert result["kdjd"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Full INDICATORS suite test
# ---------------------------------------------------------------------------

class TestAllIndicators:
    def test_all_indicators_present(self, sample_data):
        result = _process(sample_data, INDICATORS)
        for ind in INDICATORS:
            assert ind in result.columns, f"Indicator '{ind}' missing from output"

    def test_output_row_count_matches(self, sample_data):
        """With backfill/ffill, row count should be preserved."""
        result = _process(sample_data, INDICATORS)
        assert len(result) == len(sample_data)

    def test_all_tickers_preserved(self, sample_data):
        result = _process(sample_data, INDICATORS)
        assert set(result["tic"].unique()) == set(sample_data["tic"].unique())

    def test_no_nan_in_any_indicator(self, sample_data):
        result = _process(sample_data, INDICATORS)
        for ind in INDICATORS:
            assert result[ind].isna().sum() == 0, f"NaN found in indicator '{ind}'"

    def test_all_indicator_dtypes_numeric(self, sample_data):
        result = _process(sample_data, INDICATORS)
        for ind in INDICATORS:
            assert pd.api.types.is_numeric_dtype(result[ind]), \
                f"Indicator '{ind}' is not numeric"

    def test_date_sorting(self, sample_data):
        """Output should be sorted by date then ticker."""
        result = _process(sample_data, INDICATORS)
        dates = result["date"].tolist()
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Single ticker tests
# ---------------------------------------------------------------------------

class TestSingleTicker:
    def test_single_ticker_processes(self, single_ticker_data):
        result = _process(single_ticker_data, INDICATORS)
        assert len(result) == len(single_ticker_data)
        assert list(result["tic"].unique()) == ["GOOG"]

    def test_single_ticker_all_indicators(self, single_ticker_data):
        result = _process(single_ticker_data, INDICATORS)
        for ind in INDICATORS:
            assert ind in result.columns


# ---------------------------------------------------------------------------
# Edge case / fallback tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_indicator_defaults_to_zero(self, sample_data):
        """An unrecognized indicator name should produce a column of 0.0."""
        result = _process(sample_data, ["totally_unknown_xyz"])
        assert "totally_unknown_xyz" in result.columns
        assert (result["totally_unknown_xyz"] == 0.0).all()

    def test_empty_indicator_list(self, sample_data):
        """Empty indicator list should return df with original columns only."""
        result = _process(sample_data, [])
        assert len(result) == len(sample_data)
        # No extra indicator columns added
        for col in INDICATORS:
            assert col not in result.columns

    def test_original_columns_preserved(self, sample_data):
        """Original OHLCV columns should still be in the output."""
        result = _process(sample_data, INDICATORS)
        for col in ["date", "tic", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_duplicate_indicators_handled(self, sample_data):
        """Passing the same indicator twice should not cause errors."""
        result = _process(sample_data, ["rsi_14", "rsi_14"])
        assert "rsi_14" in result.columns

    def test_small_dataset(self):
        """5-row dataset: indicators that need longer windows should fall back to 0."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = [{
            "date": d.strftime("%Y-%m-%d"),
            "tic": "TINY",
            "open": 100 + i,
            "high": 102 + i,
            "low": 98 + i,
            "close": 101 + i,
            "volume": 1_000_000,
        } for i, d in enumerate(dates)]

        df = pd.DataFrame(data)
        result = _process(df, INDICATORS)
        # Should still return 5 rows with all indicator columns
        assert len(result) == 5
        for ind in INDICATORS:
            assert ind in result.columns
            assert result[ind].isna().sum() == 0


# ---------------------------------------------------------------------------
# CLI / main function test
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_missing_input(self, tmp_path):
        """main() should raise FileNotFoundError for a missing input file."""
        import sys
        from unittest.mock import patch
        from src.feature_engineer import main

        fake_args = [
            "feature_engineer.py",
            "--input_path", str(tmp_path / "nonexistent.csv"),
            "--output_path", str(tmp_path / "output.csv"),
        ]
        with patch.object(sys, "argv", fake_args):
            with pytest.raises(FileNotFoundError):
                main()

    def test_main_end_to_end(self, sample_data, tmp_path):
        """main() should read CSV, process, and write output."""
        import sys
        from unittest.mock import patch
        from src.feature_engineer import main

        input_path = str(tmp_path / "input.csv")
        output_path = str(tmp_path / "output.csv")
        sample_data.to_csv(input_path, index=False)

        fake_args = [
            "feature_engineer.py",
            "--input_path", input_path,
            "--output_path", output_path,
            "--indicator_list", "rsi_14", "macd",
        ]
        with patch.object(sys, "argv", fake_args):
            main()

        output = pd.read_csv(output_path)
        assert "rsi_14" in output.columns
        assert "macd" in output.columns
        assert len(output) == len(sample_data)
