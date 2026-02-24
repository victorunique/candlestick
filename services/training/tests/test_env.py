import pytest
import numpy as np
from src.trading_env import TradingEnv

class MockPipeline:
    def __init__(self):
        self.prices = [100, 105, 95, 100]

    def get_features(self, step_idx):
        return np.random.rand(256).astype(np.float32)
        
    def get_future_close(self, step_idx):
        return self.prices[step_idx + 1] if step_idx + 1 < len(self.prices) else self.prices[step_idx]

def test_trading_env_rewards():
    pipeline = MockPipeline()
    # Prices: 100 -> 105 (UP), 105 -> 95 (DOWN), 95 -> 100 (UP)
    
    env = TradingEnv(pipeline=pipeline, max_steps=4)
    obs, info = env.reset()
    assert obs.shape == (256,)
    
    # Step 0: 100 -> 105
    obs, reward, terminated, truncated, info = env.step(2) # 2 is UP
    assert reward == 1.0
    assert not terminated
    
    # Step 1: 105 -> 95
    obs, reward, terminated, truncated, info = env.step(2) # 2 is UP (incorrect)
    assert reward == -1.0
    
    # Step 2: 95 -> 100
    obs, reward, terminated, truncated, info = env.step(0) # 0 is SAME (incorrect)
    assert reward == -1.0 
    assert terminated
