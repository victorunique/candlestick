import os
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.history_collector import collect_day, run_cycle


@pytest.fixture
def sample_1m_data():
    """Minimal 1-minute OHLCV dataframe returned by DataFetcher."""
    dates = pd.date_range("2026-03-06 09:30", periods=3, freq="min", tz="America/New_York")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d %H:%M:%S%z").str[:-2] + ":" + dates.strftime("%z").str[-2:],
        "open": [100.0, 101.0, 102.0],
        "high": [100.5, 101.5, 102.5],
        "low": [99.5, 100.5, 101.5],
        "close": [100.2, 101.2, 102.2],
        "volume": [1000, 2000, 3000],
        "tic": ["aapl", "aapl", "aapl"],
    })


class TestCollectDay:
    """Tests for the collect_day function."""

    def test_skip_existing_file(self, tmp_path):
        """When the CSV already exists, fetch_data must not be called."""
        target = tmp_path / "AAPL_2026-03-06_1m.csv"
        target.write_text("date,open,high,low,close,volume,tic\n")

        with patch("src.history_collector.DataFetcher") as mock_cls:
            collect_day("AAPL", date(2026, 3, 6), str(tmp_path))
            mock_cls.assert_not_called()

    def test_fetch_creates_file(self, tmp_path, sample_1m_data):
        """When the CSV is missing, fetch_data is called and the file is kept."""
        target = tmp_path / "AAPL_2026-03-06_1m.csv"

        with patch("src.history_collector.DataFetcher") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_data.return_value = sample_1m_data
            mock_cls.return_value = mock_instance

            collect_day("AAPL", date(2026, 3, 6), str(tmp_path))

            mock_cls.assert_called_once_with(
                start_date="2026-03-06",
                end_date="2026-03-07",
                ticker_list=["AAPL"],
                interval="1m",
            )
            mock_instance.fetch_data.assert_called_once_with(
                output_path=str(target)
            )

    def test_no_file_on_empty_data(self, tmp_path):
        """When fetch_data raises ValueError (no data), no CSV must be created."""
        target = tmp_path / "AAPL_2026-03-07_1m.csv"

        with patch("src.history_collector.DataFetcher") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_data.side_effect = ValueError("no data is fetched.")
            mock_cls.return_value = mock_instance

            collect_day("AAPL", date(2026, 3, 7), str(tmp_path))

        assert not target.exists()

    def test_continues_on_exception(self, tmp_path, sample_1m_data):
        """A generic exception on one ticker must not prevent others from running."""
        with patch("src.history_collector.DataFetcher") as mock_cls:
            mock_instance = MagicMock()
            # First call raises, second succeeds
            mock_instance.fetch_data.side_effect = [
                RuntimeError("network error"),
                sample_1m_data,
            ]
            mock_cls.return_value = mock_instance

            # Should not raise — second ticker still processed
            run_cycle(
                tickers=["FAIL", "AAPL"],
                data_dir=str(tmp_path),
                lookback_days=1,
            )

            assert mock_instance.fetch_data.call_count == 2

    def test_filename_format(self, tmp_path, sample_1m_data):
        """Output filename must match <TICKER>_<YYYY-MM-DD>_1m.csv."""
        with patch("src.history_collector.DataFetcher") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_data.return_value = sample_1m_data
            mock_cls.return_value = mock_instance

            collect_day("TSLA", date(2026, 1, 15), str(tmp_path))

            expected_path = str(tmp_path / "TSLA_2026-01-15_1m.csv")
            mock_instance.fetch_data.assert_called_once_with(
                output_path=expected_path
            )


class TestRunCycle:
    """Tests for the run_cycle function."""

    def test_run_cycle_multiple_tickers(self, tmp_path):
        """run_cycle must call collect_day once per ticker for each day."""
        with patch("src.history_collector.collect_day") as mock_collect:
            run_cycle(
                tickers=["AAPL", "TSLA"],
                data_dir=str(tmp_path),
                lookback_days=1,
            )

            assert mock_collect.call_count == 2
            tickers_called = [call.args[0] for call in mock_collect.call_args_list]
            assert "AAPL" in tickers_called
            assert "TSLA" in tickers_called

    def test_lookback_days(self, tmp_path):
        """With lookback_days=3, run_cycle generates 3 dates and calls collect_day for each."""
        today = date.today()

        with patch("src.history_collector.collect_day") as mock_collect:
            run_cycle(
                tickers=["AAPL"],
                data_dir=str(tmp_path),
                lookback_days=3,
            )

            # 1 ticker × 3 days = 3 calls
            assert mock_collect.call_count == 3

            dates_called = sorted([call.args[1] for call in mock_collect.call_args_list])
            expected_dates = [today - timedelta(days=2), today - timedelta(days=1), today]
            assert dates_called == expected_dates
