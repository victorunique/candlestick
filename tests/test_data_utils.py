"""Tests for src.data_utils – ticker timestamp alignment."""

import pandas as pd
import pytest

from src.data_utils import align_ticker_timestamps


# ---------------------------------------------------------------------------
# align_ticker_timestamps
# ---------------------------------------------------------------------------

class TestAlignTickerTimestamps:
    """Verify that unbalanced multi-ticker data is filtered to the intersection."""

    def _make_balanced_df(self):
        """Two tickers sharing the exact same 3 timestamps."""
        data = []
        for ts in ["2024-01-01 09:30", "2024-01-01 09:31", "2024-01-01 09:32"]:
            for tic in ["AAPL", "MSFT"]:
                data.append({
                    "date": ts, "tic": tic,
                    "open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000,
                })
        return pd.DataFrame(data)

    def _make_unbalanced_df(self):
        """AAPL has 4 timestamps, MSFT only has 3 (missing 09:33).
        
        This simulates real-world intraday data where tickers have slightly
        different bar availability due to halts or delayed quotes.
        """
        data = []
        aapl_ts = ["2024-01-01 09:30", "2024-01-01 09:31", "2024-01-01 09:32", "2024-01-01 09:33"]
        msft_ts = ["2024-01-01 09:30", "2024-01-01 09:31", "2024-01-01 09:32"]

        for ts in aapl_ts:
            data.append({
                "date": ts, "tic": "AAPL",
                "open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000,
            })
        for ts in msft_ts:
            data.append({
                "date": ts, "tic": "MSFT",
                "open": 200, "high": 210, "low": 190, "close": 200, "volume": 2000,
            })
        return pd.DataFrame(data)

    def test_balanced_data_unchanged(self):
        """Balanced panel passes through with same row count."""
        df = self._make_balanced_df()
        result = align_ticker_timestamps(df)
        assert len(result) == len(df)

    def test_unbalanced_data_filtered_to_intersection(self):
        """Unbalanced data is trimmed to only common timestamps."""
        df = self._make_unbalanced_df()
        result = align_ticker_timestamps(df)
        # Common timestamps are 3 (09:30, 09:31, 09:32), so 3 × 2 tickers = 6 rows
        assert len(result) == 6
        # The extra AAPL 09:33 row should be gone
        assert "2024-01-01 09:33" not in result["date"].values

    def test_both_tickers_have_same_timestamps_after_alignment(self):
        """After alignment, both tickers must share identical timestamp sets."""
        df = self._make_unbalanced_df()
        result = align_ticker_timestamps(df)
        aapl_ts = set(result.loc[result["tic"] == "AAPL", "date"])
        msft_ts = set(result.loc[result["tic"] == "MSFT", "date"])
        assert aapl_ts == msft_ts

    def test_single_ticker_unchanged(self):
        """Single-ticker data should pass through unmodified."""
        data = [
            {"date": "2024-01-01 09:30", "tic": "SPY", "close": 100},
            {"date": "2024-01-01 09:31", "tic": "SPY", "close": 101},
        ]
        df = pd.DataFrame(data)
        result = align_ticker_timestamps(df)
        assert len(result) == 2

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame(columns=["date", "tic", "close"])
        result = align_ticker_timestamps(df)
        assert len(result) == 0

    def test_three_tickers_intersection(self):
        """With 3 tickers, only timestamps present in ALL three are kept."""
        data = [
            # All three share 09:30
            {"date": "09:30", "tic": "A", "close": 1},
            {"date": "09:30", "tic": "B", "close": 2},
            {"date": "09:30", "tic": "C", "close": 3},
            # Only A and B share 09:31
            {"date": "09:31", "tic": "A", "close": 1},
            {"date": "09:31", "tic": "B", "close": 2},
            # Only A has 09:32
            {"date": "09:32", "tic": "A", "close": 1},
        ]
        df = pd.DataFrame(data)
        result = align_ticker_timestamps(df)
        # Only 09:30 is common to all three → 3 rows
        assert len(result) == 3
        assert set(result["date"].unique()) == {"09:30"}
