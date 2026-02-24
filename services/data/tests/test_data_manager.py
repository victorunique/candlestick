import pytest
import pandas as pd
import numpy as np
import datetime
import os
from src.data_manager import DataManager

@pytest.fixture
def mock_ohlcv_data():
    dates = pd.date_range("2026-01-01", periods=2000, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "BTC/USD",
        "open": np.random.rand(2000),
        "high": np.random.rand(2000),
        "low": np.random.rand(2000),
        "close": np.random.rand(2000),
        "volume": np.random.rand(2000) * 100
    })
    return df

@pytest.fixture
def data_manager():
    # Use in-memory duckdb for testing
    db_path = ":memory:"
    manager = DataManager(db_path=db_path)
    manager.setup_schema()
    yield manager

def test_ingest_data(data_manager, mock_ohlcv_data):
    data_manager.ingest_dataframe(mock_ohlcv_data)
    
    # Check if data was written
    count = data_manager.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'BTC/USD'").fetchone()[0]
    assert count == 2000

def test_get_sliding_window(data_manager, mock_ohlcv_data):
    data_manager.ingest_dataframe(mock_ohlcv_data)
    
    # Get a timestamp somewhere in the middle
    target_time = mock_ohlcv_data.iloc[1500]["timestamp"]
    
    window = data_manager.get_sliding_window("BTC/USD", target_time, window_size=1000)
    
    # Assert window length
    assert len(window) == 1000
    
    # Assert the last row is exactly the target_time row
    assert window.iloc[-1]["timestamp"] == target_time

def test_get_sliding_window_not_enough_data(data_manager, mock_ohlcv_data):
    data_manager.ingest_dataframe(mock_ohlcv_data)
    
    # Request a window of 1000, but from index 500
    target_time = mock_ohlcv_data.iloc[500]["timestamp"]
    
    with pytest.raises(ValueError, match="Not enough data points"):
        data_manager.get_sliding_window("BTC/USD", target_time, window_size=1000)

def test_impute_missing_values(data_manager):
    # Create data with missing rows (gaps in time)
    dates = pd.date_range("2026-01-01 10:00:00", periods=5, freq="1min")
    # Drop index 2
    dates = dates.drop(dates[2])
    
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "BTC/USD",
        "open": [1.0, 1.1, 1.3, 1.4],
        "high": [1.0, 1.1, 1.3, 1.4],
        "low": [1.0, 1.1, 1.3, 1.4],
        "close": [1.0, 1.1, 1.3, 1.4], # The closing price for the missing tick should carry forward 1.1
        "volume": [10, 15, 20, 25]
    })
    
    # Ingest and fill gaps
    cleaned_df = data_manager.clean_and_impute(df)
    assert len(cleaned_df) == 5
    # The missing row (index 2) should have a volume of 0 and close price rolling over
    assert cleaned_df.iloc[2]["volume"] == 0
    assert cleaned_df.iloc[2]["close"] == 1.1
    assert cleaned_df.iloc[2]["open"] == 1.1
