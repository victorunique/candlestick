import pytest
import pandas as pd
import os
import tempfile
from src.train_ppo import train_ppo
from src.backtest import backtest

@pytest.fixture
def sample_preprocessed_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = []
    
    for i, date in enumerate(dates):
        # Generate dummy data for two assets
        for tic in ["AAPL", "MSFT"]:
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": tic,
                "open": 100 + i,
                "high": 105 + i,
                "low": 95 + i,
                "close": 100 + i,
                "volume": 1000,
                "macd": 1,
                "rsi_30": 50,
                "cci_30": 100,
                "dx_30": 20
            })
            
    df = pd.DataFrame(data)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    return df
    
def test_backtest(sample_preprocessed_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        # First, train a small model to get valid .zip and .pkl
        model_name = "test_backtest_model"
        
        train_ppo(
            df=sample_preprocessed_data,
            total_timesteps=10,
            model_dir=temp_dir,
            model_name=model_name,
            indicators=["macd", "rsi_30", "cci_30", "dx_30"],
            window_size=10
        )
        
        # Now run backtest
        model_path = os.path.join(temp_dir, model_name)
        
        # Create output dir for backtest results
        results_dir = os.path.join(temp_dir, "results")
        
        account_df, actions_df = backtest(
            df=sample_preprocessed_data, # Use same data for simplicity in test
            model_path=model_path,
            results_dir=results_dir,
            indicators=["macd", "rsi_30", "cci_30", "dx_30"],
            window_size=10
        )
        
        # Verify execution and output formats
        assert not account_df.empty
        assert "total_assets" in account_df.columns
        assert not actions_df.empty
        
        # Verify files were saved
        assert os.path.exists(os.path.join(results_dir, f"{model_name}_account_history.csv"))
        assert os.path.exists(os.path.join(results_dir, f"{model_name}_action_history.csv"))
