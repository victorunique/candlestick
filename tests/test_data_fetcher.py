import os
import pandas as pd
import pytest
from unittest.mock import patch
from src.data_fetcher import DataFetcher

@pytest.fixture
def sample_yfinance_data():
    dates = pd.date_range("2020-08-03", periods=3)
    df = pd.DataFrame({
        "Open": [10.0, 11.0, 12.0],
        "High": [10.5, 11.5, 12.5],
        "Low": [9.5, 10.5, 11.5],
        "Close": [10.2, 11.2, 12.2],
        "Adj Close": [10.2, 11.2, 12.2],
        "Volume": [100, 200, 300],
    }, index=dates)
    return df

@pytest.fixture
def sample_yfinance_data_needs_adjust():
    dates = pd.date_range("2020-08-03", periods=3)
    df = pd.DataFrame({
        "Open": [100.0, 50.0, 55.0],
        "High": [105.0, 55.0, 60.0],
        "Low": [95.0, 48.0, 52.0],
        "Close": [100.0, 50.0, 55.0],
        "Adj Close": [50.0, 50.0, 55.0],  # 2:1 split on day 2
        "Volume": [1000, 2000, 3000],
    }, index=dates)
    return df

def test_fetch_data_formatting(sample_yfinance_data, tmp_path):
    output_path = tmp_path / "test_output.csv"
    fetcher = DataFetcher(start_date="2020-08-03", end_date="2020-08-06", ticker_list=["AAPL", "MSFT"])
    
    with patch("yfinance.download") as mock_download:
        mock_download.return_value = sample_yfinance_data.copy()
        
        df = fetcher.fetch_data(output_path=str(output_path), auto_adjust=True)
        
        assert mock_download.call_count == 2
        # Dataframe should have length 6 (3 days * 2 tickers)
        assert len(df) == 6
        
        # Check standard columns
        expected_cols = ["date", "open", "high", "low", "close", "volume", "tic"]
        assert list(df.columns) == expected_cols
        
        # Check date format 'YYYY-MM-DD HH:MM:SS-TZ'
        # Example format: '2020-08-03 09:30:00-04:00' (we will test if it matches structure)
        import re
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2}$")
        assert df["date"].str.match(date_pattern).all()
        
        # Check lower case ticker
        assert set(df["tic"]) == {"aapl", "msft"}
        
        # Check CSV was written
        assert os.path.exists(output_path)
        csv_df = pd.read_csv(output_path)
        assert list(csv_df.columns) == expected_cols

def test_auto_adjust_logic(sample_yfinance_data_needs_adjust, tmp_path):
    fetcher = DataFetcher(start_date="2020-08-03", end_date="2020-08-06", ticker_list=["AAPL"])
    
    with patch("yfinance.download") as mock_download:
        mock_download.return_value = sample_yfinance_data_needs_adjust.copy()
        
        # without auto_adjust = False, it should manually adjust using Close and Adj Close
        df = fetcher.fetch_data(output_path=str(tmp_path / "test.csv"), auto_adjust=False)
        
        # Adj ratio = Adj Close / Close
        # First row: 50.0 / 100.0 = 0.5. Open 100 * 0.5 = 50.0
        assert df.iloc[0]["open"] == 50.0
        assert df.iloc[0]["close"] == 50.0
        
def test_drop_missing_data(tmp_path):
    dates = pd.date_range("2020-08-03", periods=3)
    df = pd.DataFrame({
        "Open": [10.0, None, 12.0],
        "High": [10.5, 11.5, 12.5],
        "Low": [9.5, 10.5, 11.5],
        "Close": [10.2, 11.2, 12.2],
        "Adj Close": [10.2, 11.2, 12.2],
        "Volume": [100, 200, 300],
    }, index=dates)

    fetcher = DataFetcher(start_date="2020-08-03", end_date="2020-08-06", ticker_list=["AAPL"])
    with patch("yfinance.download") as mock_download:
        mock_download.return_value = df
        res_df = fetcher.fetch_data(output_path=str(tmp_path / "test.csv"), auto_adjust=True)
        assert len(res_df) == 2  # One row dropped because of NaN in Open

def test_sorting(sample_yfinance_data, tmp_path):
    # Pass tickers in opposite order
    fetcher = DataFetcher(start_date="2020-08-03", end_date="2020-08-06", ticker_list=["ZBRA", "AAPL"])
    
    with patch("yfinance.download") as mock_download:
        mock_download.return_value = sample_yfinance_data.copy()
        
        df = fetcher.fetch_data(output_path=str(tmp_path / "test.csv"), auto_adjust=True)
        # Should be sorted chronologically and then by ticker
        assert df.iloc[0]["tic"] == "aapl"
        assert df.iloc[1]["tic"] == "zbra"

        # Check dates
        assert df.iloc[0]["date"] == df.iloc[1]["date"] # Same day
        assert df.iloc[0]["date"] < df.iloc[2]["date"] # Next day
