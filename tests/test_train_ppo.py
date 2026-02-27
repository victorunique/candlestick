import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from stable_baselines3 import PPO
from src.train_ppo import train_ppo

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
    
def test_train_ppo(sample_preprocessed_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        val_df = sample_preprocessed_data.copy()
        
        model_name = "test_ppo_model"
        
        # Train for very few timesteps just to verify the pipeline doesn't crash
        model, env_normalized = train_ppo(
            df=sample_preprocessed_data,
            total_timesteps=10,
            model_dir=temp_dir,
            model_name=model_name,
            indicators=["macd", "rsi_30", "cci_30", "dx_30"],
            window_size=10,
            ent_coef=0.01,
            learning_rate=0.00025,
            gamma=0.99
        )
        
        assert isinstance(model, PPO)
        
        # Check that files were saved
        assert os.path.exists(os.path.join(temp_dir, f"{model_name}.zip"))
        assert os.path.exists(os.path.join(temp_dir, f"{model_name}_vecnormalize.pkl"))
