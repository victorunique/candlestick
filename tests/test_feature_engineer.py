import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.feature_engineer import FeatureEngineer

@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = []
    
    # Generate mock price data for two tickers
    for tic in ["AAPL", "MSFT"]:
        base_price = 150 if tic == "AAPL" else 300
        # Random walk for prices
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
                "volume": int(np.random.uniform(1000000, 5000000))
            })
    
    df = pd.DataFrame(data)
    # Sort by date then tic
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    return df

def test_feature_engineer_init():
    indicators = ['macd', 'rsi_30']
    fe = FeatureEngineer(indicators)
    assert fe.indicators == indicators

def test_feature_engineer_preprocess(sample_data):
    indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']
    fe = FeatureEngineer(indicators)
    processed_df = fe.preprocess_data(sample_data)
    
    # Check if indicators are added
    for ind in indicators:
        assert ind in processed_df.columns
        
    # Check if shapes align (might lose some rows due to indicator windows, but handled per ticker)
    assert len(processed_df) <= len(sample_data)
    
    # AAPL and MSFT should still be present
    assert set(processed_df["tic"].unique()) == {"AAPL", "MSFT"}

def test_feature_engineer_missing_columns(sample_data):
    indicators = ['macd']
    fe = FeatureEngineer(indicators)
    bad_data = sample_data.drop(columns=["high"])
    with pytest.raises(ValueError):
        fe.preprocess_data(bad_data)

def test_feature_engineer_cli():
    # We will test the CLI directly or via main blocks if needed
    pass
